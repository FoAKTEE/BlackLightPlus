# render.jl — two-phase render driver (blacklight main.cpp role): phase 1 traces
# and caches geodesics, phase 2 integrates radiative transfer along them.
# Cross-module composition happens here at root scope, keeping Radiation free of
# a hard dependency on Simulation.

using StaticArrays
using .Radiation: eight_coefficients, polarized_march, unpolarized_march

"Vertical (image-β) direction at the camera in BH coordinates: ∂(image_to_bh)/∂β."
function _camera_up(cam::PlaneParallelCamera)
    i = deg2rad(cam.inc_deg)
    return SVector(0.0, -cos(i), 0.0, sin(i))
end

"Load an AthenaK snapshot by extension: `.bin` via `athenak_bin_snapshot` (needs γ for press→eint), else the HDF5 `.athdf` loader."
_load_athenak(path::AbstractString, γ) =
    endswith(path, ".bin") ? athenak_bin_snapshot(path; gamma=γ) : AthenaKSnapshot(path)

"""
Shared phase-1 setup: spacetime → camera → threaded geodesics → snapshot →
code-unit scale → per-ray primitive sampler. `spacetime = nothing` selects
`KerrSchildCartesian(cfg.bh_a)`; an alternative metric (e.g. `SuperposedKerrSchild`)
is dispatched on by camera, tracer, marches, and tetrads. `snapshot` may be a path
or a pre-built `AthenaKSnapshot` (avoids re-reading a large dump per camera angle).
"""
function _pipeline_setup(cfg::RunConfig, snapshot; C::Float64=80.0, spacetime=nothing)
    st = spacetime === nothing ? KerrSchildCartesian(cfg.bh_a) : spacetime
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    states, dls = trace_rays_sampled(st, rays; C=C, nmax=cfg.max_steps,
                                     escape=1.1 * cfg.camera.distance)
    snap = snapshot isa AthenaKSnapshot ? snapshot : _load_athenak(snapshot, cfg.plasma.gamma)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    γ = cfg.plasma.gamma
    sampler = x -> begin
        s = sample_primitives(snap, x)
        (rho=s.rho, pgas=pgas_from_eint(s.eint, γ), uu=s.uu, bb=s.bb, inside=s.inside)
    end
    return (st=st, cam=cam, rays=rays, states=states, dls=dls, u=u, sampler=sampler)
end

"Map `cfg.plasma` onto a `SimCoeffConfig` (shared by the full-transfer entry points); caller-specific fields are threaded through `kw...`."
function _scc_config(cfg::RunConfig, u; kw...)
    p = cfg.plasma
    return SimCoeffConfig(; bh_a=cfg.bh_a, d_unit=u.dens_unit,
                          e_unit=u.dens_unit * Constants.CL^2, b_unit=u.B_unit,
                          plasma_mu=p.mu, plasma_ne_ni=p.ne_ni,
                          plasma_use_p=p.use_p ? 1 : 0,
                          plasma_rat_low=p.rat_low, plasma_rat_high=p.rat_high,
                          plasma_gamma=p.gamma, plasma_gamma_i=p.gamma_i,
                          plasma_gamma_e=p.gamma_e, plasma_thermal_frac=p.thermal_frac,
                          plasma_power_frac=p.power_frac,
                          cut_sigma_max=p.sigma_cut, cut_theta_e_min=p.theta_e_min,
                          kw...)
end

"""
Fully-prepared coefficient config: `_scc_config` plus a one-time host precompute
of the nonthermal constants (`precompute_power_law!` when `power_frac != 0`,
`precompute_kappa!` when `kappa_frac != 0`); thermal decks skip both. Single
source for `render`, `render_amr`, and the device parameter tuples that mirror
these constants to the GPU coefficient kernels.
"""
function _prepared_scc(cfg::RunConfig, u)
    p = cfg.plasma
    scc = _scc_config(cfg, u; plasma_kappa_frac=p.kappa_frac, plasma_p=p.power_p,
                      plasma_kappa=p.kappa, plasma_w=p.w)
    p.power_frac != 0.0 && precompute_power_law!(scc, p.gamma_min, p.gamma_max)
    p.kappa_frac != 0.0 && precompute_kappa!(scc)
    return scc
end

"""
    render(cfg::RunConfig, snapshot; polarized=cfg.polarized, C=80.0, spacetime=nothing)
        -> NamedTuple (I, Q, U, V) — each npix×npix (Q/U/V ≡ 0 for unpolarized)

Full pipeline on an AthenaK snapshot (`.athdf`/`.bin` path or pre-built
`AthenaKSnapshot`): camera → threaded geodesics → per-ray fluid sampling +
invariant coefficients → transfer march → images. `spacetime` (default
`KerrSchildCartesian(cfg.bh_a)`) injects an alternative metric, e.g. a static
`SuperposedKerrSchild` superposition (frozen orbital phase, no puncture boost).
`C` is the sampling density (step h = (r − r_h)/C); raise it (C ≥ 160) to
reduce null-constraint drift on plunging rays.
"""
function render(cfg::RunConfig, snapshot; polarized::Bool=cfg.polarized,
                C::Float64=80.0, spacetime=nothing)
    # `:analytic` exact-Kerr tracer: one equatorial landing point per ray
    # (no ODE, no GRMHD snapshot); `snapshot` is ignored on this path.
    if cfg.tracer === :analytic
        r = render_analytic(cfg)
        return (I=r.I, Q=zero(r.I), U=zero(r.I), V=zero(r.I))
    end
    (; st, cam, rays, states, dls, u, sampler) =
        _pipeline_setup(cfg, snapshot; C=C, spacetime=spacetime)
    scc = _prepared_scc(cfg, u)

    n = length(rays)
    ν = cfg.frequency_hz
    Lu = u.L_unit
    up = _camera_up(cam)
    out = Matrix{Float64}(undef, 4, n)
    Threads.@threads for i in 1:n
        if polarized
            s4 = polarized_march(st, scc, sampler, states[i], dls[i], ν, Lu; cam_up=up)
            out[:, i] .= s4
        else
            out[1, i] = unpolarized_march(st, scc, sampler, states[i], dls[i], ν, Lu)
            out[2, i] = out[3, i] = out[4, i] = 0.0
        end
    end
    npx = cam.npix
    shape(k) = reshape(out[k, :], npx, npx)
    return (I=shape(1), Q=shape(2), U=shape(3), V=shape(4))
end

"""
    render_with_powerlaw(cfg, snapshot_path; p, gamma_min, gamma_max)

Nonthermal render: like `render` but with the power-law tail's host-precomputed
constants filled (`precompute_power_law!`). `cfg.plasma.power_frac` sets the
tail fraction; the thermal fraction is `cfg.plasma.thermal_frac`.
"""
function render_with_powerlaw(cfg::RunConfig, snapshot_path::AbstractString;
                              p::Float64=3.5, gamma_min::Float64=25.0, gamma_max::Float64=1.0e7)
    (; st, cam, rays, states, dls, u, sampler) = _pipeline_setup(cfg, snapshot_path)
    scc = _scc_config(cfg, u; plasma_p=p)
    precompute_power_law!(scc, gamma_min, gamma_max)
    n = length(rays)
    out = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        out[i] = unpolarized_march(st, scc, sampler, states[i], dls[i],
                                   cfg.frequency_hz, u.L_unit)
    end
    npx = cam.npix
    return (I=reshape(out, npx, npx),)
end

# Per-pixel, per-band paint of the exact-Kerr equatorial tracer, shared by
# render_analytic and photon_ring_layers: direct-invert the closed-form equatorial
# geodesic for ring order `n` at Bardeen (α, β), then evaluate the equatorial
# source seam. `srcfun` is the injected disk closure or `nothing` (r⁻² placeholder);
# returns the landing observables and painted intensity (0 off the disk). The GPU
# kernel (gpu_analytic.jl) inlines this same seam by hand — keep them in sync.
@inline function _paint_layer(α, β, θo, a, n::Integer, ro, srcfun)
    rs, φs, ts, νr, valid, λ, η = Geodesics._analytic_emission_full(α, β, θo, a, n; ro=ro)
    iv = valid ? equatorial_intensity(rs, νr, a; source=srcfun, λ=λ, η=η) : zero(rs)
    return (iv=iv, rs=rs, phis=φs, ts=ts, nu=νr, valid=valid)
end

"""
    render_analytic(cfg::RunConfig; bands=(0,1,2), ro=cfg.camera.distance, source=nothing)
        -> NamedTuple (I, Ib, rs, ts, phis, redshift_sign, bands)

The `:analytic` exact-Kerr tracer: for every camera pixel and ring order
n ∈ `bands`, direct-invert the equatorial geodesic (`analytic_emission`) and
evaluate the equatorial source there. Returns per-band observable maps — BL
source radius `rs`, time-of-flight `ts`, azimuth `phis`, `redshift_sign` (each
npix×npix, NaN off the disk) — the per-band images `Ib`, and `I = Σ_n Ib[n]`.
`source = nothing` uses the placeholder r⁻² emissivity; an `EquatorialDisk`
paints the AART g-disk redshift source with the Doppler+lensing brightness.
Camera image coordinates are Bardeen's: α_Bardeen = α_image, β_Bardeen = β_image
(validated against the numerical tracer). Threaded over pixels.
"""
function render_analytic(cfg::RunConfig; bands=(0, 1, 2), ro::Float64=cfg.camera.distance,
                         source=nothing)
    a = cfg.bh_a
    θo = deg2rad(cfg.camera.inclination_deg)
    npx = cfg.camera.resolution
    fov = cfg.camera.fov
    edges = range(-fov / 2, fov / 2; length=npx + 1)
    centres = (edges[1:end-1] .+ edges[2:end]) ./ 2
    nb = length(bands)
    # per-band observables; column-major (M[iβ, iα]) matches render()'s reshape order
    rs = [fill(NaN, npx, npx) for _ in 1:nb]
    ts = [fill(NaN, npx, npx) for _ in 1:nb]
    phis = [fill(NaN, npx, npx) for _ in 1:nb]
    rsign = [fill(NaN, npx, npx) for _ in 1:nb]
    Ib = [zeros(npx, npx) for _ in 1:nb]
    I = zeros(npx, npx)
    # inject the disk source as a closure; Geodesics stays Radiation-agnostic
    srcfun = source === nothing ? nothing :
             (r_s, νr, aa, λ, η) -> Radiation.equatorial_disk_intensity(source, r_s, νr, aa, λ, η)
    Threads.@threads for iα in 1:npx
        αB = centres[iα]               # camera image-α == Bardeen α
        for iβ in 1:npx
            βB = centres[iβ]
            acc = 0.0
            for (bi, n) in enumerate(bands)
                p = _paint_layer(αB, βB, θo, a, n, ro, srcfun)
                if p.valid
                    rs[bi][iβ, iα] = p.rs
                    ts[bi][iβ, iα] = p.ts
                    phis[bi][iβ, iα] = p.phis
                    rsign[bi][iβ, iα] = p.nu
                    Ib[bi][iβ, iα] = p.iv
                    acc += p.iv
                end
            end
            I[iβ, iα] = acc
        end
    end
    return (I=I, Ib=Ib, rs=rs, ts=ts, phis=phis, redshift_sign=rsign, bands=collect(bands))
end

"""
    photon_ring_layers(cfg::RunConfig; disk, bands=(0,1,2), dx=0.02, box=12.0,
                       npts=100, ro=cfg.camera.distance, fudge=1.0)
        -> NamedTuple (I, Ib, masks, grid, N, box, dx, bands)

Layered photon-ring render of the equatorial disk (AART's band loop): for each
ring order n ∈ `bands`, build its lensing band on the shared Bardeen (α,β) grid
over `[-box, box]²` at spacing `dx`, direct-invert `analytic_emission` on the
in-band pixels only, and paint `equatorial_disk_intensity`. Restricting to the
band mask is what keeps the n ≥ 1 rings razor-thin (the `valid` flag alone
over-counts high-n landings). Returns the per-band images `Ib[n]` (N×N, rows =
β) and `masks[n]`, the composite `I = Σ_n fudge^{[n>0]} Ib[n]`, and the shared
grid metadata; all layers share `N`/coordinates so they sum directly. Threaded.
"""
function photon_ring_layers(cfg::RunConfig; disk::Radiation.EquatorialDisk,
                            bands=(0, 1, 2), dx::Float64=0.02, box::Float64=12.0,
                            npts::Int=100, ro::Float64=cfg.camera.distance, fudge::Float64=1.0)
    a = cfg.bh_a
    θo = deg2rad(cfg.camera.inclination_deg)
    nb = length(bands)
    inner0, outer0 = lensing_band(a, θo, 0; npts=npts, box=box)
    grid, _, N = band_grid(inner0, outer0; dx=dx, box=box)
    npix = N * N
    Ib = [zeros(N, N) for _ in 1:nb]
    # Matrix{Bool}, not BitMatrix: threaded scattered writes to a BitArray are a
    # read-modify-write race on shared 64-bit chunks that silently drops bits.
    masks = [fill(false, N, N) for _ in 1:nb]
    # paint through the same equatorial_intensity seam render_analytic uses
    srcfun = (r_s, νr, aa, λ, η) -> Radiation.equatorial_disk_intensity(disk, r_s, νr, aa, λ, η)
    for (bi, n) in enumerate(bands)
        inner, outer = lensing_band(a, θo, n; npts=npts, box=box)
        _, bmask, _ = band_grid(inner, outer; dx=dx, box=box)
        img = Ib[bi]
        mk = masks[bi]
        Threads.@threads for idx in 1:npix
            bmask[idx] || continue
            α = grid[idx, 1]
            β = grid[idx, 2]
            p = _paint_layer(α, β, θo, a, n, ro, srcfun)
            p.valid || continue
            ix = ((idx - 1) % N) + 1
            iy = ((idx - 1) ÷ N) + 1
            img[iy, ix] = p.iv
            mk[iy, ix] = p.iv > 0
        end
    end
    I = zeros(N, N)
    for (bi, n) in enumerate(bands)
        I .+= (n == 0 ? 1.0 : fudge) .* Ib[bi]
    end
    return (I=I, Ib=Ib, masks=masks, grid=grid, N=N, box=box, dx=dx, bands=collect(bands))
end

"Diagnostic: ratio of polarized-march I (with alpha_Q=alpha_V=rho=0 forced) to the
unpolarized-march I on the same rays — isolates coherency-march bookkeeping (expect ~1)."
function polunpol_consistency(cfg::RunConfig, snapshot_path::AbstractString)
    (; st, cam, rays, states, dls, u, sampler) = _pipeline_setup(cfg, snapshot_path)
    p = cfg.plasma
    # deliberately minimal scc: with αQ=αV=ρ forced to 0, only the β-ratio and
    # cut fields affect the ratio — the emission-model fields are irrelevant
    scc = SimCoeffConfig(bh_a=cfg.bh_a, d_unit=u.dens_unit,
                         e_unit=u.dens_unit * Constants.CL^2, b_unit=u.B_unit,
                         plasma_rat_low=p.rat_low, plasma_rat_high=p.rat_high,
                         plasma_gamma=p.gamma, cut_sigma_max=p.sigma_cut, cut_theta_e_min=p.theta_e_min)
    n = length(rays)
    ipol = zeros(n); iunp = zeros(n)
    Threads.@threads for i in 1:n
        ipol[i] = Radiation.polarized_march(st, scc, sampler, states[i], dls[i],
                                            cfg.frequency_hz, u.L_unit;
                                            cam_up=_camera_up(cam), intensity_only=true)[1]
        iunp[i] = unpolarized_march(st, scc, sampler, states[i], dls[i],
                                    cfg.frequency_hz, u.L_unit)
    end
    return sum(ipol) / sum(iunp)
end
