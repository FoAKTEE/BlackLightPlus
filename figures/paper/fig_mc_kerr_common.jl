# fig_mc_kerr_common.jl — definitions shared by the Monte Carlo generators of the Kerr
# accretion flow (fig_mc_kerr.jl and the bias and convergence studies; included, never run alone).
#
# The plasma is that of Fig. polarized: the Kerr accretion flow of AthenaK snapshot 00999 on
# Kerr a = 0.9, scaled to M87* (black-hole mass 6.5e9 Msun at 16.8 Mpc, the mass scale MASS_SCALE
# at which the 230 GHz flux density seen from i = 17 deg is 0.5 Jy), R_low = 1, R_high = 10,
# adiabatic index 13/9 with total-pressure matching, sigma cut 1, no Theta_e floor, step
# control C = 80.
#
# Two descriptions of the same cells:
#   the snapshot   trilinear interpolation between cell centres (the images of the paper);
#   CellCentred    every point takes the values of the cell that contains it, the plasma
#                  whose cells the Monte Carlo emits from their centres.
# Ray-traced spectra (`rt_nuLnu`) are the isotropic-equivalent nu L_nu = 4 pi nu sum(I dA)
# of StokesI `render` images of the plasma rotated about the spin axis (`RotatedGrid`, so
# the camera, fixed at azimuth 0, views it from any azimuth): the sum of the nested square
# fields NEST (full field of view in rg, pixels per side), each with the next field cut
# out, so the whole domain is imaged and the inner flow at the resolution of the innermost
# field.
using Printf
using Statistics
using StaticArrays
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, ME
using BlackLightPlus.Matter: GridProvider, PlasmaPrimitives
using BlackLightPlus.Spacetime: ks_radius
using BlackLightPlus.Data: locate, write_npy
using BlackLightPlus.Execution: Spectrum
import BlackLightPlus.Matter: primitives, domain_radius
import BlackLightPlus.Products: zone_blocks
include(joinpath(@__DIR__, "m87_scaling.jl"))
using .M87Scaling

const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const ROOT = dirname(dirname(@__DIR__))
const OUT = get(ENV, "BLP_OUT", joinpath(@__DIR__, "out", "mc_kerr"))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

const SPIN = 0.9
const M_BH = M87_MASS_G                            # the M87* scaling of every GRMHD model
const L_UNIT = GNEWT * M_BH / CL^2                 # rg in cm
# g: the mass scale (density unit times rg^3) at which the 230 GHz flux density seen from
# i = 17 deg at 16.8 Mpc is 0.5 Jy, the calibration of every GRMHD model of the paper
# (m87_scaling.jl, run 00999)
const MASS_SCALE = haskey(ENV, "BLP_MC_MASS_SCALE") ? parse(Float64, ENV["BLP_MC_MASS_SCALE"]) :
                   m87_density_scale("00999")
const MEC2 = ME * CL * CL
const NEST = ((1920.0, 240), (480.0, 240), (40.0, 240), (20.0, 320))
const BANDS = (17.0, 50.0, 90.0)
const HALF = 10.0
const SPEC = (spec_nu_min = 1.0e8, spec_nu_max = 1.0e24, n_ebins = 160)

kerr_config(; nu = 2.3e11, inc = 60.0, fov = 30.0, res = 64, C = 80.0) =
    RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = nu,
              camera = CameraConfig(inclination_deg = inc, distance = 1000.0, fov = fov,
                                    resolution = res),
              plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                    use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
              max_steps = round(Int, 2500 * C / 80), C = C, escape_factor = 1.1)

"""
    CellCentred(snap)

The snapshot `snap` sampled piecewise constant: a point takes the primitives at the centre
of the cell of the finest block that contains it (`locate`), the cell values that the
Monte Carlo zones emit with. Its emission zones are those of the snapshot.
"""
struct CellCentred{S} <: GridProvider
    s::S
end
@inline function primitives(p::CellCentred, x::SVector{4})
    b = p.s.blocks
    mb = locate(b, x[2], x[3], x[4])
    mb == 0 && return primitives(p.s, x)
    c(a, n) = b.x_left[a, mb] +
              clamp(floor((x[a + 1] - b.x_left[a, mb]) / b.dx[a, mb] + 0.5), 0, n - 1) * b.dx[a, mb]
    return primitives(p.s, SVector(x[1], c(1, b.ni), c(2, b.nj), c(3, b.nk)))
end
domain_radius(p::CellCentred) = domain_radius(p.s)
zone_blocks(p::CellCentred) = p.s.blocks

"""
    RotatedGrid(inner, st; phi = 0, rcut = Inf)

The grid `inner` rotated by `phi` about the spin axis and emptied beyond Kerr–Schild
radius `rcut` of `st`: primitives(R, x) = R(phi) primitives(inner, R(-phi) x) for the
Cartesian Kerr–Schild vectors (velocity, field), a symmetry of the Kerr metric.
"""
struct RotatedGrid{P,S} <: GridProvider
    inner::P
    st::S
    c::Float64
    s::Float64
    rcut::Float64
end
RotatedGrid(inner, st; phi = 0.0, rcut = Inf) =
    RotatedGrid(inner, st, cos(phi), sin(phi), Float64(rcut))

@inline function primitives(r::RotatedGrid, x::SVector{4})
    ks_radius(r.st, x) > r.rcut && return PlasmaPrimitives{Float64}()
    X = x[2]; Y = x[3]
    p = primitives(r.inner, SVector(x[1], r.c * X + r.s * Y, -r.s * X + r.c * Y, x[4]))
    p.inside || return p
    rot(v) = SVector(r.c * v[1] - r.s * v[2], r.s * v[1] + r.c * v[2], v[3])
    return PlasmaPrimitives{Float64}(p.rho, rot(p.uu), p.eint, rot(p.bb), true)
end
domain_radius(r::RotatedGrid) = min(r.rcut, domain_radius(r.inner))

"""
    rt_nuLnu(prov, nu, inc, phi; nest = NEST, C = 80) -> Float64

Ray-traced isotropic-equivalent nu L_nu (erg/s) of the plasma `prov` seen from inclination
`inc` (deg) and azimuth `phi` (rad): the nested fields `nest`, each with the next one cut
out (the last whole).
"""
function rt_nuLnu(prov, nu, inc, phi; nest = NEST, C = 80.0)
    st = KerrSchildCartesian(SPIN)
    tot = 0.0
    for k in eachindex(nest)
        fov, res = nest[k]
        plan = prepare(kerr_config(; nu, inc, fov, res, C); spacetime = st,
                       matter = RotatedGrid(prov, st; phi), mode = StokesI())
        I = render(plan; backend = :cpu).I
        s = sum(I)
        if k < lastindex(nest)
            n_in = round(Int, res * nest[k + 1][1] / fov)
            i0 = (res - n_in) ÷ 2
            s -= sum(@view I[i0+1:i0+n_in, i0+1:i0+n_in])
        end
        tot += s * (fov / res * L_UNIT)^2
    end
    return 4π * nu * tot
end

"Gauss–Legendre nodes and weights on [-1, 1] (n = 2, 3, 4)."
gauss_legendre(n) = n == 2 ? ([-1 / sqrt(3), 1 / sqrt(3)], [1.0, 1.0]) :
                    n == 3 ? ([-sqrt(0.6), 0.0, sqrt(0.6)], [5 / 9, 8 / 9, 5 / 9]) :
                    n == 4 ? ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
                              [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]) :
                    error("gauss_legendre: n = $n")

"""
    rt_band(prov, nu, inc_c; nth = 2, nphi = 8, shift = 0) -> (mean, values)

The ray-traced nu L_nu averaged over the inclination band inc_c ± HALF (Gauss–Legendre in
cos i, `nth` nodes) and `nphi` uniform azimuths (offset by `shift` of a spacing): the band
mean of the isotropic-equivalent luminosity that the Monte Carlo band collects.
"""
function rt_band(prov, nu, inc_c; nth = 2, nphi = 8, shift = 0.0, kw...)
    lo = cosd(inc_c + HALF); hi = cosd(inc_c - HALF)
    xs, ws = gauss_legendre(nth)
    acc = 0.0
    vals = Float64[]
    for (x, w) in zip(xs, ws)
        inc = acosd(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        for k in 0:nphi-1
            v = rt_nuLnu(prov, nu, inc, 2π * (k + shift) / nphi; kw...)
            push!(vals, v)
            acc += 0.5 * w * v / nphi
        end
    end
    return acc, vals
end

"""
    row_stats(spec::Spectrum, j) -> (L, sigma, neff, count)

Row `j` of a spectrum as the isotropic-equivalent nu L_nu (erg/s) over its solid angle,
its standard error from the per-bin sum of (w E)^2, the effective photon number
(sum w E)^2 / sum (w E)^2 and the photon count.
"""
function row_stats(s::Spectrum, j::Integer)
    f = MEC2 / s.dlnu * (4π / s.dOmega[j])
    a = s.sumwE[j, :]; a2 = s.sumwE2[j, :]
    neff = [q2 > 0 ? q * q / q2 : 0.0 for (q, q2) in zip(a, a2)]
    return f .* a, f .* sqrt.(a2), neff, Float64.(s.count[j, :])
end

"""
    stats_table(res) -> Matrix

Columns: nu, then for each scattering order n = 0..max_scatt and then the sum over orders,
for each band row (BANDS, then the whole sky): L, sigma, neff, count. Without scattering
only the sum (= order 0).
"""
function stats_table(res)
    sp = res.orders
    nu = BlackLightPlus.Execution.nu_centers(res.total)
    cols = Any[nu]
    total = sp[1]
    specs = length(sp) > 1 ? [sp..., nothing] : [nothing]
    for s in specs
        for j in 1:(length(BANDS) + 1)
            if s === nothing
                # the sum over orders: the bands of the result and its sky row
                ss = j <= length(BANDS) ? res.spectra[j] : res.total
                append!(cols, row_stats(ss, 1))
            else
                append!(cols, row_stats(s, j))
            end
        end
    end
    return hcat(cols...)
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Symbol) = json_str(string(x))
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Nothing) = "null"
json_str(x::Union{AbstractVector,Tuple}) = "[" * join(json_str.(x), ", ") * "]"
json_str(x::NamedTuple) = json_str(Dict(string(k) => v for (k, v) in pairs(x)))
function json_str(d::AbstractDict)
    items = [json_str(string(k)) * ": " * json_str(d[k]) for k in sort!(collect(keys(d)))]
    return "{" * join(items, ", ") * "}"
end
write_json(path, d) = open(io -> write(io, json_str(d), "\n"), path, "w")

"The run ledger of a GRMHD result as a Dict."
ledger(r) = Dict{String,Any}("n_zones" => r.n_zones, "n_gen" => r.n_gen, "n_recorded" => r.n_recorded,
                 "n_children" => r.n_children, "emitted_power" => r.emitted_power,
                 "escaped_power" => r.escaped_power, "absorbed_power" => r.absorbed_power,
                 "captured_power" => r.captured_power, "dropped_power" => r.dropped_power,
                 "w_primary" => r.w_primary, "w_children" => r.w_children,
                 "w_unspawned" => r.w_unspawned, "scatter_loss" => r.scatter_loss,
                 "seconds" => r.run.seconds)
lg(a...) = (println(a...); flush(stdout))

# ---------------------------------------------------------------- emission within the cells
using BlackLightPlus.Data: emission_sites, zone_tables
using BlackLightPlus.MonteCarlo: CurvedMarch, EMISSIVITY_TABLE, GenerationTables, generate,
                                 mean_weight
using BlackLightPlus.Matter: ZoneSite, emission_zone, sample
using BlackLightPlus.MonteCarlo: Superphoton, j_nu
using BlackLightPlus.Numerics: HPL
using LinearAlgebra: dot, det
using BlackLightPlus.Numerics: MCRng
using BlackLightPlus.Numerics: widest_layout, mc_uniform
using BlackLightPlus.Execution: InclinationBands, ExactBins, band, resolve
using BlackLightPlus.Products: GRMHDSpectrumResult
import BlackLightPlus.Products: emit
const PR = BlackLightPlus.Products

"""
    InCellSource(zones, dx)

The zone source `zones` (`Products.ZoneSource`) with every primary emitted at a point `x`
drawn uniformly in its cell (`dx`, the cell size of each zone; three uniforms of the photon's
own stream), with the fluid frame of the plasma sampled at `x`: the emission of the
cell-centred plasma, in which every point of a cell carries the cell's primitives decoded
with the metric at the point. The frequency and the direction are drawn from the emissivity
of the cell centre (the zone tables), and the weight is multiplied by
`j_ν(x)/j_ν(centre)` at the photon's frequency and pitch angle, so the expected emission is
that of the point; a point where the plasma is cut emits a photon of weight 0 (drawn at the
valid point `vp` of its cell, the centre for every zone of `Data.emission_sites`).
"""
struct InCellSource{Z}
    inner::Z
    dx::Vector{SVector{3,Float64}}
    vp::Vector{SVector{4,Float64}}
end
@inline function emit(s::InCellSource, item, rng)
    zs = s.inner
    z = searchsortedfirst(zs.offset, item) - 1
    site = zs.sites[z]
    u1, rng = mc_uniform(rng); u2, rng = mc_uniform(rng); u3, rng = mc_uniform(rng)
    d = s.dx[z]
    x = site.x + SVector(0.0, (u1 - 0.5) * d[1], (u2 - 0.5) * d[2], (u3 - 0.5) * d[3])
    loc, _ = sample(zs.provider, zs.st, x, zs.params)
    ok = loc.valid && loc.n_e > 0.0 && loc.theta_e > 0.0
    # a cut point emits nothing: its photon, drawn at the cell centre for a finite momentum,
    # carries weight 0
    zone = emission_zone(zs.provider, zs.st,
                         ZoneSite(ok ? x : s.vp[z], site.dV, site.n_e, site.theta_e, site.B), zs.params)
    sp, rng = generate(zone, GenerationTables(zs.weights, zs.dnmax[z], zs.emissivity), rng)
    r = 0.0
    if ok
        eps = -dot(sp.p, zone.ucon)
        bb = sqrt(abs(dot(zone.bcon, zone.gcov * zone.bcon)))
        cth = bb > 0.0 ? clamp(dot(sp.p, zone.bcon) / (bb * eps), -1.0, 1.0) : 0.0
        sth = sqrt(max(0.0, 1.0 - cth * cth))
        nu = eps * MEC2 / HPL
        jc = j_nu(site.n_e, site.theta_e, site.B, nu, sth)
        r = jc > 0.0 ? j_nu(loc.n_e, loc.theta_e, loc.B, nu, sth) / jc : 0.0
        isfinite(r) || (r = 0.0)
    end
    out = Superphoton(sp.x, sp.p, sp.w * r; tau_abs = 0.0, tau_scatt = 0.0, nscatt = 0,
                      x1i = sp.x1i, x2i = sp.x2i, ne0 = sp.ne0, thetae0 = sp.thetae0, b0 = sp.b0)
    return out, rng
end

"""
    partial_sites(blocks, st, params, provider; r_max = 10, npts = 16) -> (sites, points)

The cells within Kerr–Schild radius `r_max` whose centre is cut (the plasma decoded with the
metric at the centre fails the cuts) but which hold valid points, the cell's primitives
decoded with the metric at the point passing them: next to the horizon the decode changes
across a cell, and these cells emit in the cell-centred plasma. Each becomes a zone at its
centre with the whole cell's proper volume and the plasma of the first valid point among
`npts` uniform points (a fixed stream per block); `points` holds that point. The in-cell
source weights every photon by the emissivity at its point over the zone's, so the valid part
of the cell emits exactly and the cut part not at all.
"""
function partial_sites(b, st, params, provider; r_max = 10.0, npts = 16)
    per = Vector{Vector{Tuple{ZoneSite{Float64},SVector{4,Float64}}}}(undef, b.nmb)
    Lu3 = params.units.L_unit^3
    Threads.@threads for mb in 1:b.nmb
        out = Tuple{ZoneSite{Float64},SVector{4,Float64}}[]
        rng = MCRng(0x7a7a, mb)
        d = SVector(b.dx[1, mb], b.dx[2, mb], b.dx[3, mb])
        for k in 1:b.nk, j in 1:b.nj, i in 1:b.ni
            x = SVector(0.0, b.x_left[1, mb] + (i - 1) * d[1], b.x_left[2, mb] + (j - 1) * d[2],
                        b.x_left[3, mb] + (k - 1) * d[3])
            dist, r = BlackLightPlus.Spacetime.proxy_distance(st, x)
            (r <= r_max && dist >= BlackLightPlus.Spacetime.HORIZON_EPS_SAMPLED) || continue
            locate(b, x[2], x[3], x[4]) == mb || continue
            sc, g = sample(provider, st, x, params)
            (sc.valid && sc.n_e > 0.0 && sc.theta_e > 0.0) && continue
            for _ in 1:npts
                u1, rng = mc_uniform(rng); u2, rng = mc_uniform(rng); u3, rng = mc_uniform(rng)
                y = x + SVector(0.0, (u1 - 0.5) * d[1], (u2 - 0.5) * d[2], (u3 - 0.5) * d[3])
                sy, _ = sample(provider, st, y, params)
                if sy.valid && sy.n_e > 0.0 && sy.theta_e > 0.0
                    detg = det(g[1])
                    dV = (detg < 0.0 ? sqrt(-detg) : 1.0) * d[1] * d[2] * d[3] * Lu3
                    push!(out, (ZoneSite{Float64}(x, dV, sy.n_e, sy.theta_e, sy.B), y))
                    break
                end
            end
        end
        per[mb] = out
    end
    all = reduce(vcat, per)
    return [a for (a, _) in all], [p for (_, p) in all]
end

"""
    mc_incell(plan, snap, Ns; seed, scattering, bias, bias_order, max_scatt) -> GRMHDSpectrumResult

`mc_spectrum(plan, Ns; …)` of the Kerr deck (stride 1, generation 1e8-1e16 Hz, the bins SPEC
and the bands BANDS ± HALF) with every primary emitted uniformly within its cell
([`InCellSource`](@ref)); the plan's matter is the plasma of the transport, the zones are the
cells of `snap` (`Data.emission_sites`) and the cells next to the horizon whose centre is cut
but which hold valid points ([`partial_sites`](@ref)).
"""
function mc_incell(plan, snap, Ns; seed = 1, scattering = false, bias = 1.0e4, bias_order = 1.0,
                   max_scatt = 3)
    t0 = time_ns()
    st = plan.spacetime
    K = scattering ? max_scatt : 0
    useed = UInt32(seed & 0xffffffff)
    sites = emission_sites(snap.blocks, st, plan.params; stride = 1, seed = useed)
    vp = [s_.x for s_ in sites]
    xs, xp = partial_sites(snap.blocks, st, plan.params, plan.provider)
    sites = vcat(sites, xs); vp = vcat(vp, xp)
    wgt, Nsz, dnmax = zone_tables(sites, Ns / (log(1.0e16) - log(1.0e8)); nu_min = 1.0e8, nu_max = 1.0e16)
    b = snap.blocks
    dx = Vector{SVector{3,Float64}}(undef, length(sites))
    Threads.@threads for i in eachindex(sites)
        s = sites[i]
        mb = locate(b, s.x[2], s.x[3], s.x[4])
        dx[i] = SVector(b.dx[1, mb], b.dx[2, mb], b.dx[3, mb])
    end
    offset = PR.zone_offsets(Nsz, useed)
    N = offset[end]
    R_em = max(100.0, maximum(z -> 1.2 * BlackLightPlus.Spacetime.proxy_distance(st, z.x)[2], sites))
    ctl = CurvedMarch(st; C = 80.0, escape = 1.0e3, R_em = R_em)
    o = (table = scattering ? BlackLightPlus.Plans.hotcross_table(plan) : nothing,
         bias = (b = bias, b_max = PR.MC_BIAS_MAX, split = 1, order = Float64(bias_order)), K = K)
    tr = PR.MCTransport(ctl, st, plan.provider, PR.mc_opacities(plan.params, o, mean_weight(wgt)),
                        Val(true), Val(scattering))
    rows = InclinationBands(BANDS, HALF)
    orders = ntuple(_ -> ExactBins(rows; nu_min = SPEC.spec_nu_min, nu_max = SPEC.spec_nu_max,
                                   n_ebins = SPEC.n_ebins), K + 1)
    ledger = PR.mc_ledger()
    exec = resolve(:cpu)
    src = InCellSource(PR.ZoneSource(sites, plan.provider, st, plan.params, wgt, EMISSIVITY_TABLE,
                                     dnmax, offset), dx, vp)
    nch = PR.run_layers(exec, src, tr, useed, N, orders, ledger, widest_layout(max(1, N), K),
                        scattering, nothing)
    total = similar(orders[1])
    foreach(q -> merge!(total, q), orders)
    spec = Spectrum(total)
    nb = length(rows.lo)
    rec = PR.run_record(exec, plan, t0)
    return GRMHDSpectrumResult(collect(rows.incl_deg), rows.half_width_deg, spec.dOmega[1:nb],
                               [band(spec, k) for k in 1:nb], band(spec, nb + 1),
                               [Spectrum(q) for q in orders], K, length(sites), N,
                               PR.ledger_count(ledger, PR.ESCAPED), nch,
                               PR.ledger_totals(ledger)..., scattering, plan, rec)
end
