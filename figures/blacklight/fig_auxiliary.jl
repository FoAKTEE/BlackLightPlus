# fig_auxiliary.jl — auxiliary images and false-colour renderings (paper figures 11-14)
# from the fiducial deck's traced rays sampled along their paths:
#   aux    — per pixel of the fiducial camera: the elapsed Kerr-Schild time from the
#            ray's end to the camera, the proper length ∫(g_ij dx^i dx^j)^½, the affine
#            parameter, I_j = ∫ j_I dλ, τ = ∫ α_I dλ, and the electron temperature
#            averaged by affine parameter, by emission and by optical depth
#            (d⟨q⟩/dτ = q - ⟨q⟩ integrated source → camera);
#   edge   — renderings at θ = 70°: width 60 M with matter inside r = 45 M, and width
#            24 M with matter inside r = 20 M and z < -r_h; each in flat spacetime and in
#            Kerr; features: dense plasma blue with τ_s = 1/30 M⁻¹, σ > 1 red with
#            τ_s = 1/25 M⁻¹ (no σ cut), β⁻¹ = 1 crossings green with opacity 0.25;
#   face   — face-on (θ = 0) with matter inside r = 10 M: dense plasma (τ_s = 1/5), the
#            230 GHz image, Θ_e > THETA_HOT (τ_s = 1).
# The paper's density thresholds (5e-18 and 2e-17 g cm⁻³) belong to its own density
# scale; here "dense" is ρ > ρ_max/20 (edge) and ρ > ρ_max/5 (face) with ρ_max the dump's
# largest density in cgs, and the hot threshold is THETA_HOT = 50 (the paper: 20), since the
# torus electrons inside r = 10 M are everywhere above 20.
# Colours are mixed in CIE XYZ (c += (c_i - c)(1 - e^{-τ_s ds}) inside a region; a
# crossing blends c ← (1 - a) c + a c_i) and written as XYZ planes; the plotter projects
# to sRGB. Writes out/blacklight/auxiliary/*.npy and record.json.
# Run from the repository root:   julia -t 32 --project=. figures/blacklight/fig_auxiliary.jl
include(joinpath(@__DIR__, "common.jl"))
using BlackLightPlus.Spacetime: horizon_radius

const XYZ_BLUE = SVector(0.1805, 0.0722, 0.9505)
const XYZ_RED = SVector(0.4124, 0.2126, 0.0193)
const XYZ_GREEN = SVector(0.3576, 0.7152, 0.1192)
const THETA_HOT = 50.0

# aux maps of a plan: each a npix × npix matrix
function aux_maps(plan)
    traj = trace(plan; backend = :cpu)
    n = plan.camera.npix
    names = (:time, :length, :lambda, :Ij, :tau, :theta_lambda, :theta_j, :theta_tau)
    maps = Dict(k => zeros(n, n) for k in names)
    L = plan.params.units.L_unit; ν = plan.params.nu_obs
    Threads.@threads for px in 1:n^2
        s = sampled_path(plan, traj.states[px], traj.dls[px])
        m = length(s.t)
        ib, ia = (px - 1) % n + 1, (px - 1) ÷ n + 1
        maps[:time][ib, ia] = s.t[end] - s.t[1]                   # camera time minus the end time
        maps[:length][ib, ia] = sum(s.ds)
        lam = sum(s.dlam)
        maps[:lambda][ib, ia] = lam
        Ij = 0.0; tau = 0.0; thl = 0.0; thj = 0.0; tht = 0.0
        for k in 1:m-1                                            # source → camera
            dl = s.dlam[k] * L / ν
            s.valid[k] || continue
            Ij += s.j[k] * dl
            dτ = s.alpha[k] * dl
            tau += dτ
            thl += s.theta_e[k] * s.dlam[k]
            thj += s.theta_e[k] * s.j[k] * dl
            tht += (s.theta_e[k] - tht) * (1 - exp(-dτ))
        end
        maps[:Ij][ib, ia] = Ij; maps[:tau][ib, ia] = tau
        maps[:theta_lambda][ib, ia] = lam > 0 ? thl / lam : 0.0
        maps[:theta_j][ib, ia] = Ij > 0 ? thj / Ij : 0.0
        maps[:theta_tau][ib, ia] = tht
    end
    return maps
end

# rendering: `fills` are (predicate(s, k), colour, tau_s per M), `crossings` are
# (value(s, k), threshold, colour, opacity); `inside(x)` restricts the matter
function render_maps(plan, fills, crossings, inside)
    traj = trace(plan; backend = :cpu)
    n = plan.camera.npix
    XYZ = zeros(n, n, 3)
    Threads.@threads for px in 1:n^2
        s = sampled_path(plan, traj.states[px], traj.dls[px])
        c = zero(SVector{3,Float64})
        m = length(s.t)
        prevs = [NaN for _ in crossings]
        for k in 1:m-1                                            # source → camera
            ok = s.valid[k] && inside(s.x[k])
            if ok
                for (pred, col, ts) in fills
                    pred(s, k) && (c = c + (col - c) * (1 - exp(-ts * s.ds[k])))
                end
            end
            for (ci, (val, thr, col, op)) in enumerate(crossings)
                v = ok ? val(s, k) : NaN
                if isfinite(v) && isfinite(prevs[ci]) && (v - thr) * (prevs[ci] - thr) < 0
                    c = (1 - op) * c + op * col
                end
                prevs[ci] = v
            end
        end
        XYZ[(px - 1) % n + 1, (px - 1) ÷ n + 1, :] .= c
    end
    return XYZ
end

function main()
    out = outdir("auxiliary")
    snap = AthenaKSnapshot(dump_path())
    kerr = KerrSchildCartesian(BL_SPIN)
    r_h = horizon_radius(kerr)
    rec = Dict{String,Any}()

    # figures 11-12
    t0 = time()
    maps = aux_maps(prepare(fiducial_config(); spacetime = kerr, matter = snap, mode = StokesI()))
    for (k, v) in maps
        write_npy(joinpath(out, "aux_$(k).npy"), v)
    end
    rec["aux_seconds"] = time() - t0
    logf("aux maps in %.0f s: time %.1f-%.1f, length %.1f-%.1f, tau max %.2f\n", rec["aux_seconds"],
         minimum(maps[:time]), maximum(maps[:time]), minimum(maps[:length]), maximum(maps[:length]), maximum(maps[:tau]))

    # figure 13: edge-on renderings (no sigma cut)
    nocut = PlasmaConfig(sigma_cut = -1.0)
    rho_max = Float64(maximum(@view snap.blocks.data[1, :, :, :, :])) * prepare(fiducial_config(); spacetime = kerr, matter = snap, mode = StokesI()).params.units.dens_unit
    rec["rho_max_cgs"] = rho_max
    rho_edge = rho_max / 20; rho_face = rho_max / 5
    rec["rho_thresholds_cgs"] = Dict("edge" => rho_edge, "face" => rho_face); rec["theta_hot"] = THETA_HOT
    logf("rho_max %.3e g/cm^3: thresholds edge %.3e, face %.3e\n", rho_max, rho_edge, rho_face)
    rho_fill = ((s, k) -> s.rho[k] > rho_edge, XYZ_BLUE, 1 / 30)
    sig_fill = ((s, k) -> s.sigma[k] > 1.0, XYZ_RED, 1 / 25)
    beta_cross = ((s, k) -> s.beta_inv[k], 1.0, XYZ_GREEN, 0.25)
    t0 = time()
    for (tag, fov, rmax, zcut) in (("wide", 60.0, 45.0, false), ("zoom", 24.0, 20.0, true))
        inside = x -> (norm(x) < rmax) && (!zcut || x[3] < -r_h)
        for (stag, stc) in (("flat", Minkowski()), ("kerr", kerr))
            plan = prepare(fiducial_config(; inc = 70.0, fov = fov, plasma = nocut); spacetime = stc, matter = snap, mode = StokesI())
            XYZ = render_maps(plan, (rho_fill, sig_fill), (beta_cross,), inside)
            write_npy(joinpath(out, "edge_$(tag)_$(stag)_X.npy"), XYZ[:, :, 1])
            write_npy(joinpath(out, "edge_$(tag)_$(stag)_Y.npy"), XYZ[:, :, 2])
            write_npy(joinpath(out, "edge_$(tag)_$(stag)_Z.npy"), XYZ[:, :, 3])
        end
    end
    rec["edge_seconds"] = time() - t0
    logf("edge renderings in %.0f s\n", rec["edge_seconds"])

    # figure 14: face-on renderings and image, matter inside r = 10
    t0 = time()
    inside10 = x -> norm(x) < 10.0
    plan = prepare(fiducial_config(; inc = 0.0); spacetime = kerr, matter = snap, mode = StokesI())
    XYZ = render_maps(plan, (((s, k) -> s.rho[k] > rho_face, XYZ_BLUE, 1 / 5),), (), inside10)
    for (c, nm) in enumerate(("X", "Y", "Z")); write_npy(joinpath(out, "face_rho_$nm.npy"), XYZ[:, :, c]); end
    XYZ = render_maps(plan, (((s, k) -> s.theta_e[k] > THETA_HOT, XYZ_RED, 1.0),), (), inside10)
    for (c, nm) in enumerate(("X", "Y", "Z")); write_npy(joinpath(out, "face_theta_$nm.npy"), XYZ[:, :, c]); end
    inner = MaskedProvider(snap, (x, prim) -> (x[2]^2 + x[3]^2 + x[4]^2) < 100.0)
    img = render(prepare(fiducial_config(; inc = 0.0); spacetime = kerr, matter = inner, mode = StokesI()); backend = :cpu)
    write_npy(joinpath(out, "face_I.npy"), img.I)
    rec["face_seconds"] = time() - t0
    rec["face_flux_jy"] = flux_jy(img.I, BL_FOV, BL_RES)
    logf("face-on renderings in %.0f s, image %.3f Jy\n", rec["face_seconds"], rec["face_flux_jy"])
    write_json(joinpath(out, "record.json"), record("auxiliary"; result = rec))
end

struct MaskedProvider{P,F} <: GridProvider
    p::P
    keep::F
end
BlackLightPlus.Matter.primitives(m::MaskedProvider, x::SVector{4}) = begin
    prim = primitives(m.p, x)
    (prim.inside && m.keep(x, prim)) ? prim : PlasmaPrimitives{typeof(prim.rho)}()
end
BlackLightPlus.Matter.domain_radius(m::MaskedProvider) = domain_radius(m.p)

main()
