# Monte Carlo — grmhd_model.jl
#
# GRMHD emission model: feeds the Monte Carlo generate -> track -> record loop
# with an AthenaK GRMHD snapshot in Cartesian Kerr-Schild spacetime (after
# grmonty harm_utils.c init_weight_table/init_zone/get_fluid_zone). The fluid
# is discretised into `EmissionZone`s, one per emitting cell, each carrying its
# cgs plasma state and proper volume dV = sqrt(-g) dx^3 L_unit^3. `plasma_at`
# reuses the imaging pipeline's fluid decode verbatim (same sampler, electron
# model and cuts), so the MC sees bit-identical plasma to the polarized imager.

# O(1) leaf-block locator, used for finest-owner de-duplication in `emission_zones`.
using ..Simulation: meshblock_index

# ================================================================ GRMHDModel

"""
    GRMHDModel

The GRMHD Monte Carlo emission model: an AthenaK snapshot + a Cartesian
Kerr-Schild spacetime + the cgs unit system and plasma configuration. NOT
`isbits` (it wraps the snapshot arrays and the mutable `SimCoeffConfig`) — it is
the HOST description consumed by `emission_zones` (to build the generation tables)
and `plasma_at` (sampled along each superphoton geodesic during tracking).

Fields
- `spacetime` : `KerrSchildCartesian(a)`.
- `snap`      : `AthenaKSnapshot` (the fluid primitives + AMR geometry).
- `cfg`       : `SimCoeffConfig` — units (`d_unit`/`e_unit`/`b_unit`), the
                ti_te_beta electron model (`rat_high`/`rat_low`/`use_p`/`gamma`s) and
                the plasma cuts (`cut_sigma_max`, `cut_theta_e_min`, ...).
- `gamma`     : adiabatic index used for `p_gas = (gamma-1) eint`.
- `L_unit`    : GM/c^2 [cm] — the code-length -> cm scale (fixes the optical-depth
                geometric factor and the proper cell volume).
- `r_h`       : outer horizon KS radius (capture surface).
- `R_em`      : emission-confinement radius [M]; a photon outbound past this streams
                freely to infinity (its escape direction is frozen — see the tracker).
"""
struct GRMHDModel{ST,SN,C}
    spacetime::ST
    snap::SN
    cfg::C
    gamma::Float64
    L_unit::Float64
    r_h::Float64
    R_em::Float64
end

"""
    GRMHDModel(snapshot_path; bh_a, m_bh_cgs, mass_scale_cgs, rat_high=10, rat_low=1,
               use_p=true, gamma=13/9, gamma_i=5/3, gamma_e=4/3, mu=0.5, ne_ni=1,
               sigma_cut=1, theta_e_min=-1, thermal_frac=1, R_em=100) -> GRMHDModel

Load an AthenaK `.athdf` snapshot and assemble the GRMHD MC model.
Units follow `Simulation.units` (`dens_unit = mass_scale/L^3`,
`e_unit = dens_unit c^2`, `B_unit = c sqrt(4 pi dens_unit)`), and the plasma /
electron model matches `render`'s `SimCoeffConfig` exactly (same ti_te_beta
`electron_theta`, same `plasma_cut_omits` sigma/theta_e gate).
"""
function GRMHDModel(snapshot_path::AbstractString;
                    bh_a::Real, m_bh_cgs::Real, mass_scale_cgs::Real,
                    rat_high::Real = 10.0, rat_low::Real = 1.0, use_p::Bool = true,
                    gamma::Real = 13.0 / 9.0, gamma_i::Real = 5.0 / 3.0,
                    gamma_e::Real = 4.0 / 3.0, mu::Real = 0.5, ne_ni::Real = 1.0,
                    sigma_cut::Real = 1.0, theta_e_min::Real = -1.0,
                    thermal_frac::Real = 1.0, R_em::Real = 100.0)
    st = KerrSchildCartesian(Float64(bh_a))
    snap = AthenaKSnapshot(snapshot_path)
    u = units(Float64(m_bh_cgs), Float64(mass_scale_cgs))
    cfg = SimCoeffConfig(bh_a = Float64(bh_a), d_unit = u.dens_unit,
                         e_unit = u.dens_unit * CL^2, b_unit = u.B_unit,
                         plasma_mu = Float64(mu), plasma_ne_ni = Float64(ne_ni),
                         plasma_use_p = use_p ? 1 : 0,
                         plasma_rat_low = Float64(rat_low),
                         plasma_rat_high = Float64(rat_high),
                         plasma_gamma = Float64(gamma), plasma_gamma_i = Float64(gamma_i),
                         plasma_gamma_e = Float64(gamma_e),
                         plasma_thermal_frac = Float64(thermal_frac),
                         cut_sigma_max = Float64(sigma_cut),
                         cut_theta_e_min = Float64(theta_e_min))
    return GRMHDModel(st, snap, cfg, Float64(gamma), u.L_unit,
                      horizon_radius(st), Float64(R_em))
end

# ================================================================ plasma_at

# Vacuum sentinel (keeps the metric so the tracker can still push the geodesic).
@inline function _vacuum(gcov, gcon)
    z = SVector(0.0, 0.0, 0.0, 0.0)
    return (n_e = 0.0, theta_e = 0.0, B = 0.0,
            ucon = z, ucov = z, bcon = z, gcov = gcov, gcon = gcon, inside = false)
end

"""
    plasma_at(model, x::SVector{4}) -> NamedTuple

Emitting-plasma state at CKS 4-position `x`, decoded EXACTLY as the imaging
`eight_coefficients` does: trilinear `sample_primitives` -> `primitive_four_vectors`
-> cgs conversion -> `electron_theta` (ti_te_beta) -> `plasma_cut_omits`. Returns
`(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon, inside)`; `inside = false` (with
zeroed plasma but valid `gcov`/`gcon`) for vacuum, off-grid, non-positive density,
or any cut-omitted (e.g. sigma > sigma_cut jet) sample. `B` is in Gauss, `n_e` in
cm^-3, `theta_e` dimensionless.
"""
function plasma_at(model::GRMHDModel, x::SVector{4})
    st = model.spacetime
    gcov = metric(st, x)
    gcon = inverse_metric(st, x)
    s = sample_primitives(model.snap, x)
    (!s.inside || s.rho <= 0.0) && return _vacuum(gcov, gcon)
    (s.bb[1] == 0.0 && s.bb[2] == 0.0 && s.bb[3] == 0.0) && return _vacuum(gcov, gcon)
    pgas = pgas_from_eint(s.eint, model.gamma)
    pgas <= 0.0 && return _vacuum(gcov, gcon)

    fv = primitive_four_vectors(s.uu[1], s.uu[2], s.uu[3],
                                s.bb[1], s.bb[2], s.bb[3], gcov, gcon)
    b_sq = fv.b_sq
    b_sq <= 0.0 && return _vacuum(gcov, gcon)

    cfg = model.cfg
    pd = plasma_decode(cfg, s.rho, pgas, b_sq)
    pd.omit && return _vacuum(gcov, gcon)

    return (n_e = pd.n_e_cgs, theta_e = pd.theta_e, B = pd.bb_cgs,
            ucon = fv.ucon, ucov = fv.ucov, bcon = fv.bcon,
            gcov = gcov, gcon = gcon, inside = true)
end

# ================================================================ emission_zones

# Finest meshblock owning CKS point (x,y,z); 0 if outside every block. Delegates
# to the reader's `meshblock_index` — the same locator `sample_primitives` uses,
# preserving the finest/last-wins tiebreak that prevents double-counting a coarse
# cell overlapped by a finer block.
@inline _owner_block(snap, x, y, z) = meshblock_index(snap, x, y, z)

"""
    emission_zones(model; max_zones=typemax(Int), stride=1, r_emit_max=Inf)
        -> Vector{EmissionZone}

Iterate the snapshot's cells and return one `EmissionZone` per emitting cell
(plasma passes `plasma_at`'s cuts, `n_e > 0`, `theta_e > 0`), with proper
volume `dV = sqrt(-g) dx1 dx2 dx3 L_unit^3`. A cell is skipped when a finer
AMR block owns its centre (de-duplication). `stride > 1` subsamples the grid,
inflating each retained cell's `dV` by `stride^3` — a fast, unbiased estimator
for tests. `r_emit_max` drops cells beyond that KS radius: the tenuous outer
atmosphere passes the cuts but emits negligibly while forcing whole-domain
tracking, so capping to the luminous torus is a large speedup at no luminosity
cost (`Inf` disables). `max_zones` caps the list (diagnostic guard).
"""
function emission_zones(model::GRMHDModel; max_zones::Integer = typemax(Int),
                        stride::Integer = 1, r_emit_max::Real = Inf)
    snap = model.snap
    st = model.spacetime
    rmax = Float64(r_emit_max)
    zones = EmissionZone[]
    s3 = Float64(stride)^3
    @inbounds for mb in 1:snap.nmb
        xl = snap.x_left[mb]
        dxb = snap.dx[mb]
        dVcode = dxb[1] * dxb[2] * dxb[3] * s3
        Lu3 = model.L_unit^3
        for k in 1:stride:snap.nk, j in 1:stride:snap.nj, i in 1:stride:snap.ni
            xx = xl[1] + (i - 1) * dxb[1]
            yy = xl[2] + (j - 1) * dxb[2]
            zz = xl[3] + (k - 1) * dxb[3]
            _owner_block(snap, xx, yy, zz) == mb || continue      # finest-block only
            xc = SVector(0.0, xx, yy, zz)
            ks_radius(st, xc) <= rmax || continue                 # luminous-torus cap
            pl = plasma_at(model, xc)
            (pl.inside && pl.n_e > 0.0 && pl.theta_e > 0.0) || continue
            g = pl.gcov
            detg = det(g)
            sqrtg = detg < 0.0 ? sqrt(-detg) : 1.0                # CKS: sqrt(-g) == 1
            dV = sqrtg * dVcode * Lu3
            push!(zones, EmissionZone(n_e = pl.n_e, theta_e = pl.theta_e, B = pl.B,
                                      dV = dV, x = xc, ucon = pl.ucon, ucov = pl.ucov,
                                      bcon = pl.bcon, gcov = g, gcon = pl.gcon))
            length(zones) >= max_zones && return zones
        end
    end
    return zones
end
