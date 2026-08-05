# Monte Carlo — emission_tables.jl
#
# Host-side generation tables (after gpumonty weights.cu `init_weight_table` /
# kernels.cu `init_zone`). The emitting plasma is a set of `EmissionZone`s; two
# frequency tables are built once: the superphoton weight w(nu) = L_nu/(h N_s),
# stored as log(Sum_zones int_jnu_zone(nu_i) dV / (h N_s)) on a log-nu grid,
# and the per-zone target count N_s = Int dln(nu) int_jnu_zone dV / (h w) with
# its rejection envelope dnmax = max_nu int_jnu_zone/w. `EmissionZone.dV` folds
# the proper-volume factor sqrt(-g) dx^3 L_unit^3 into one number.

# ---- default frequency band / resolution (gpumonty sphere_model: NUMIN 1e8) ----
const _MC_NU_MIN_DEFAULT = 1.0e8
const _MC_NU_MAX_DEFAULT = 1.0e16
const _MC_N_ESAMP_DEFAULT = 400
# Floor added to a weight before dividing / logging, so a zone with vanishing
# emissivity can never divide-by-zero or log(0) (gpumonty `WEIGHT_MIN`/1e-100).
const _WEIGHT_FLOOR = 1.0e-100

# ============================================================ EmissionZone

"""
    EmissionZone

One emitting fluid cell, the host-side input to superphoton generation (`isbits`,
StaticArrays-typed). Plasma state is cgs; the geometry is BlackLightPlus-native
(signature (-,+,+,+), covariant/contravariant `SVector{4}`s and `SMatrix{4,4}`
metrics).

Fields
- `n_e`      : electron number density [cm^-3].
- `theta_e`  : dimensionless electron temperature Theta_e = k T_e / m_e c^2.
- `B`        : magnetic-field strength [Gauss].
- `dV`       : proper emitting volume `sqrt(-g) dx^3 L_unit^3` [cm^3] — the single
               factor multiplying `int_jnu_zone` to get an emission rate.
- `x`        : coordinate position x^mu (emission point).
- `ucon`,`ucov` : fluid 4-velocity, contravariant / covariant.
- `bcon`     : magnetic-field 4-vector b^mu (its spatial direction sets b-hat).
- `gcov`,`gcon` : metric and inverse metric at `x`.
"""
struct EmissionZone
    n_e::Float64
    theta_e::Float64
    B::Float64
    dV::Float64
    x::SVector{4,Float64}
    ucon::SVector{4,Float64}
    ucov::SVector{4,Float64}
    bcon::SVector{4,Float64}
    gcov::SMatrix{4,4,Float64,16}
    gcon::SMatrix{4,4,Float64,16}
end

"Keyword constructor; widens every argument to the `isbits` field types."
function EmissionZone(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon)
    return EmissionZone(Float64(n_e), Float64(theta_e), Float64(B), Float64(dV),
                        SVector{4,Float64}(x),
                        SVector{4,Float64}(ucon), SVector{4,Float64}(ucov),
                        SVector{4,Float64}(bcon),
                        SMatrix{4,4,Float64}(gcov), SMatrix{4,4,Float64}(gcon))
end

# ============================================================ weight table

"""
    WeightTable

Precomputed superphoton-weight function `w(nu)` on a log-frequency grid. `lwgt[i]`
holds `log w` at node `nu_i = exp(lnu_min + (i-1) dlnu)`, `i = 1 … N_ESAMP+1`.
`weight_at` reads it back by log-nu interpolation. Host storage (`Vector`).
"""
struct WeightTable
    lnu_min::Float64
    lnu_max::Float64
    dlnu::Float64
    lwgt::Vector{Float64}   # log w at the N_ESAMP+1 grid nodes
end

"""
    weight_at(nu, wgt::WeightTable) -> Float64

Superphoton weight `w(nu)` [photons/packet] by piecewise log-linear
interpolation in `ln nu` (gpumonty `linear_interp_weight`), clamped to the grid
endpoints. Host wrapper over the shared `_mc_weight_at` (mc_common.jl).
"""
@inline weight_at(nu::Real, wgt::WeightTable) =
    _mc_weight_at(nu, wgt.lwgt, wgt.lnu_min, wgt.dlnu, length(wgt.lwgt))

"""
    build_weight_table(zones, Ns; n_esamp, nu_min, nu_max) -> WeightTable

Build `w(nu) = Sum_zones int_jnu_zone(nu) dV / (h Ns)` on a log-nu grid of
`n_esamp+1` nodes over `[nu_min, nu_max]`, storing `log w`. `Ns` is the target
total superphoton count (sets the packet weight; the emergent physics is
independent of it). After gpumonty `init_weight_table`.
"""
function build_weight_table(zones::AbstractVector{EmissionZone}, Ns::Real;
                            n_esamp::Integer = _MC_N_ESAMP_DEFAULT,
                            nu_min::Real = _MC_NU_MIN_DEFAULT,
                            nu_max::Real = _MC_NU_MAX_DEFAULT)
    lnu_min = log(nu_min)
    lnu_max = log(nu_max)
    dlnu = (lnu_max - lnu_min) / n_esamp
    hNs = HPL * Float64(Ns)
    lwgt = Vector{Float64}(undef, n_esamp + 1)
    @inbounds for m in 0:n_esamp
        nu = exp(m * dlnu + lnu_min)
        s = 0.0
        for z in zones
            s += int_jnu_zone(z.n_e, z.theta_e, z.B, nu) * z.dV
        end
        lwgt[m + 1] = log(s / hNs + _WEIGHT_FLOOR)
    end
    return WeightTable(lnu_min, lnu_max, dlnu, lwgt)
end

"""
    zone_nsuperphotons(zone, wgt) -> (N_s, dnmax)

Expected superphoton count and frequency-rejection envelope for `zone`:

    N_s   = (dV / h) Int dln(nu) int_jnu_zone(nu) / w(nu)
    dnmax = max_nu  int_jnu_zone(nu) / w(nu)

Both use the tabulated grid-node weights (parity with gpumonty `init_zone`).
"""
function zone_nsuperphotons(zone::EmissionZone, wgt::WeightTable)
    n_nodes = length(wgt.lwgt)
    dnmax = 0.0
    ninterp = 0.0   # Int dln(nu) int_jnu_zone / w
    @inbounds for m in 0:(n_nodes - 1)
        nu = exp(m * wgt.dlnu + wgt.lnu_min)
        w = exp(wgt.lwgt[m + 1])
        dn = int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) / (w + _WEIGHT_FLOOR)
        dnmax = max(dnmax, dn)
        ninterp += wgt.dlnu * dn
    end
    N_s = zone.dV / HPL * ninterp
    return N_s, dnmax
end

# ==================================================== cached emission table
#
# The GRMHD setup would otherwise evaluate `int_jnu_zone(zone, nu_m)` — a
# 512-node composite-Simpson pitch integral — twice per (zone, node): once in
# `build_weight_table`, once per zone in `zone_nsuperphotons`.
# `ZoneEmissionTable` computes the (n_esamp+1) x nzones values once (threaded
# over zones, race-free and deterministic) and both consumers read it; Float64
# storage keeps the fast path bit-identical to the serial recompute. The
# one-zone sphere path keeps the un-cached methods above.

"""
    ZoneEmissionTable

Cached per-(zone, frequency-node) pitch-angle-integrated emissivity for a GRMHD
emitting-zone list: `jnu[m+1, z] = int_jnu_zone(zone_z, nu_m)` on the shared
log-nu grid, `dV[z]` the zone's proper emitting volume. Built once and consumed
by both `build_weight_table` and `zone_counts`. Host storage.
"""
struct ZoneEmissionTable
    lnu_min::Float64
    lnu_max::Float64
    dlnu::Float64
    n_esamp::Int
    jnu::Matrix{Float64}    # (n_esamp+1, nzones): jnu[m+1, z] = int_jnu_zone(zone_z, nu_m)
    dV::Vector{Float64}     # per-zone proper emitting volume [cm^3]
end

"""
    build_zone_emission_table(zones; n_esamp, nu_min, nu_max) -> ZoneEmissionTable

Compute the `(n_esamp+1) x nzones` emission-integral table `jnu[m+1,z] =
int_jnu_zone(zone_z, nu_m)` once, threaded over zones. Each thread writes only
its own zone columns, so the fill is race-free and bit-identical regardless of
thread count.
"""
function build_zone_emission_table(zones::AbstractVector{EmissionZone};
                                   n_esamp::Integer = _MC_N_ESAMP_DEFAULT,
                                   nu_min::Real = _MC_NU_MIN_DEFAULT,
                                   nu_max::Real = _MC_NU_MAX_DEFAULT)
    lnu_min = log(nu_min)
    lnu_max = log(nu_max)
    dlnu = (lnu_max - lnu_min) / n_esamp
    nz = length(zones)
    jnu = Matrix{Float64}(undef, n_esamp + 1, nz)
    dV = Vector{Float64}(undef, nz)
    Threads.@threads for z in 1:nz
        zn = zones[z]
        dV[z] = zn.dV
        ne = zn.n_e; the = zn.theta_e; B = zn.B
        @inbounds for m in 0:n_esamp
            nu = exp(m * dlnu + lnu_min)
            jnu[m + 1, z] = int_jnu_zone(ne, the, B, nu)
        end
    end
    return ZoneEmissionTable(lnu_min, lnu_max, dlnu, Int(n_esamp), jnu, dV)
end

"""
    build_weight_table(jt::ZoneEmissionTable, Ns; nthreads) -> WeightTable

Fast-path weight table from a prebuilt `ZoneEmissionTable`: `w(nu_m) = (Sum_z
jt.jnu[m,z] jt.dV[z]) / (h Ns)`, stored as `log w`. Threaded over zones into
per-thread partial accumulators reduced in fixed thread-index order, so the
result is deterministic but may differ from the serial method in the last bits
(different zone summation order).
"""
function build_weight_table(jt::ZoneEmissionTable, Ns::Real;
                            nthreads::Integer = Threads.nthreads())
    n_nodes = jt.n_esamp + 1
    nz = size(jt.jnu, 2)
    hNs = HPL * Float64(Ns)
    nchunks = clamp(Int(nthreads), 1, max(1, nz))
    partials = [zeros(Float64, n_nodes) for _ in 1:nchunks]
    Threads.@threads for c in 1:nchunks
        s = partials[c]
        lo = div((c - 1) * nz, nchunks) + 1
        hi = div(c * nz, nchunks)
        @inbounds for z in lo:hi
            dVz = jt.dV[z]
            for m in 1:n_nodes
                s[m] += jt.jnu[m, z] * dVz
            end
        end
    end
    lwgt = Vector{Float64}(undef, n_nodes)
    @inbounds for m in 1:n_nodes
        s = 0.0
        for c in 1:nchunks          # fixed thread-index order -> deterministic
            s += partials[c][m]
        end
        lwgt[m] = log(s / hNs + _WEIGHT_FLOOR)
    end
    return WeightTable(jt.lnu_min, jt.lnu_max, jt.dlnu, lwgt)
end

"""
    zone_counts(jt::ZoneEmissionTable, wgt::WeightTable; nthreads)
        -> (Ns::Vector{Float64}, dnmax::Vector{Float64})

Per-zone expected superphoton count `N_s[z]` and frequency-rejection envelope
`dnmax[z]` for every zone in `jt`, read from the cached `jt.jnu`. Threaded over
zones (independent per-zone work, bit-identical regardless of thread count).
`jt` and `wgt` must share the frequency grid (asserted).
"""
function zone_counts(jt::ZoneEmissionTable, wgt::WeightTable;
                     nthreads::Integer = Threads.nthreads())
    n_nodes = jt.n_esamp + 1
    length(wgt.lwgt) == n_nodes ||
        error("zone_counts: table nodes ($n_nodes) != weight-table nodes ($(length(wgt.lwgt)))")
    nz = size(jt.jnu, 2)
    Nsv = Vector{Float64}(undef, nz)
    dnv = Vector{Float64}(undef, nz)
    Threads.@threads for z in 1:nz
        dnmax = 0.0
        ninterp = 0.0
        @inbounds for m in 1:n_nodes
            w = exp(wgt.lwgt[m])
            dn = jt.jnu[m, z] / (w + _WEIGHT_FLOOR)
            dnmax = max(dnmax, dn)
            ninterp += wgt.dlnu * dn
        end
        @inbounds Nsv[z] = jt.dV[z] / HPL * ninterp
        @inbounds dnv[z] = dnmax
    end
    return Nsv, dnv
end

# ==================================================== tabulated generation (opt-in)
#
# The frequency-rejection accept ratio in `sample_zone_photon` evaluates the
# expensive `int_jnu_zone` pitch integral once per rejection iteration; for a
# fixed zone it is a pure function of `nu`, so `IntJnuTable` caches
# `log int_jnu_zone` on a log-nu grid and the generator interpolates it like
# `weight_at`. Opt-in only (`tabulated_gen = false` default): the interpolation
# error can flip a `u2 <= accept` decision, so spectra agree within Monte Carlo
# noise but per-photon streams are not guaranteed bitwise identical.
#
# ACCEPTANCE <= 1 BY CONSTRUCTION: `tabulated_dnmax` maxes int/w over the SAME
# interpolant the accept ratio reads, on the table's own nodes. Both log int
# and log w are piecewise linear in ln nu between those nodes (every weight
# node is an int node), hence so is their log-ratio, whose supremum sits ON a
# node — interpolation can never push the accept probability above 1.
#
# CHORD CLAMP: where log int is CONCAVE in ln nu (synchrotron peak, exponential
# cutoff tail) the exact int/w ratio exceeds the weight-grid node max by the
# weight table's own interpolation error — a region where the exact path clips
# its acceptance at 1. Clamping each fine node to the coarse chord reproduces
# that clipped density, pins the envelope at the coarse-node max, and keeps the
# streams essentially identical to the exact path.

# Default refinement of the int-jnu grid over the weight grid: 10x the
# 400-interval default -> 4001 nodes, so the interpolation error is negligible
# against the per-bin Monte Carlo noise at production Ns.
const _MC_INT_JNU_REFINE = 10

"""
    IntJnuTable

Tabulated `log int_jnu_zone` for one zone on a log-nu grid: `ljnu[i]` holds
`log(int_jnu_zone(nu_i) + _WEIGHT_FLOOR)` at `nu_i = exp(lnu_min + (i-1) dlnu)`.
`ljnu` is any indexable vector (host `Vector` or a column `view` of the GRMHD
per-zone matrix). Consumed by `sample_zone_photon`'s frequency rejection when
tabulated generation is opted in.
"""
struct IntJnuTable{V<:AbstractVector{Float64}}
    lnu_min::Float64
    dlnu::Float64
    ljnu::V
end

"""
    int_jnu_at(nu, gt::IntJnuTable) -> Float64

`int_jnu_zone(nu)` by piecewise log-linear interpolation in `ln nu`, through
the shared `_mc_weight_at` (mc_common.jl). Clamped to the grid endpoints.
"""
@inline int_jnu_at(nu::Real, gt::IntJnuTable) =
    _mc_weight_at(nu, gt.ljnu, gt.lnu_min, gt.dlnu, length(gt.ljnu))

"""
    build_int_jnu_table(zone, wgt; refine=_MC_INT_JNU_REFINE) -> IntJnuTable

Tabulate `log int_jnu_zone` for `zone` on a `refine`-fold refinement of the
weight table's log-nu grid (same band; every weight node is an int node — the
alignment the acceptance<=1 envelope argument needs). Interior fine nodes store
`min(exact, coarse chord)` — the chord clamp of the section header, keeping the
rejection envelope at the coarse-node max.
"""
function build_int_jnu_table(zone::EmissionZone, wgt::WeightTable;
                             refine::Integer = _MC_INT_JNU_REFINE)
    k = Int(refine)
    n_coarse = length(wgt.lwgt) - 1
    n_fine = k * n_coarse
    dlnu = (wgt.lnu_max - wgt.lnu_min) / n_fine
    # exact log-integrand on the weight nodes: the fine grid's anchors AND the clamp.
    lc = Vector{Float64}(undef, n_coarse + 1)
    @inbounds for m in 0:n_coarse
        nu = exp(m * wgt.dlnu + wgt.lnu_min)
        lc[m + 1] = log(int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) + _WEIGHT_FLOOR)
    end
    ljnu = Vector{Float64}(undef, n_fine + 1)
    @inbounds for i in 0:n_fine
        m, r = divrem(i, k)
        if r == 0
            ljnu[i + 1] = lc[m + 1]
        else
            f = r / k
            chord = (1.0 - f) * lc[m + 1] + f * lc[m + 2]
            nu = exp(i * dlnu + wgt.lnu_min)
            lx = log(int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) + _WEIGHT_FLOOR)
            ljnu[i + 1] = min(lx, chord)          # chord clamp (see section header)
        end
    end
    return IntJnuTable(wgt.lnu_min, dlnu, ljnu)
end

"""
    tabulated_dnmax(gt, wgt) -> Float64

Frequency-rejection envelope for the tabulated accept ratio: the max of
`int_jnu_at / (weight_at + _WEIGHT_FLOOR)` over the int-table nodes, which by
piecewise linearity is the supremum over all `nu` — the tabulated acceptance is
<= 1 by construction. Replaces the exact-integrand `dnmax` when generation is
tabulated.
"""
function tabulated_dnmax(gt::IntJnuTable, wgt::WeightTable)
    dnmax = 0.0
    @inbounds for i in eachindex(gt.ljnu)
        nu = exp((i - 1) * gt.dlnu + gt.lnu_min)
        dnmax = max(dnmax, exp(gt.ljnu[i]) / (weight_at(nu, wgt) + _WEIGHT_FLOOR))
    end
    return dnmax
end

"""
    zone_int_jnu_tables(jt::ZoneEmissionTable, wgt) -> (lint, dnmax)

GRMHD twin of `build_int_jnu_table`/`tabulated_dnmax`: the per-(zone, node)
pitch integrals are already cached in `jt.jnu`, so tabulated generation only
needs their logs `lint[m, z] = log(jt.jnu[m, z] + _WEIGHT_FLOOR)` on the shared
grid (refine = 1, so the acceptance<=1 node argument holds) plus the per-zone
envelope `dnmax[z]`. Threaded over zones; `jt` and `wgt` must share the
frequency grid (asserted).
"""
function zone_int_jnu_tables(jt::ZoneEmissionTable, wgt::WeightTable)
    n_nodes = jt.n_esamp + 1
    length(wgt.lwgt) == n_nodes ||
        error("zone_int_jnu_tables: table nodes ($n_nodes) != weight-table nodes ($(length(wgt.lwgt)))")
    nz = size(jt.jnu, 2)
    lint = Matrix{Float64}(undef, n_nodes, nz)
    dnv = Vector{Float64}(undef, nz)
    Threads.@threads for z in 1:nz
        @inbounds for m in 1:n_nodes
            lint[m, z] = log(jt.jnu[m, z] + _WEIGHT_FLOOR)
        end
        dnv[z] = tabulated_dnmax(IntJnuTable(jt.lnu_min, jt.dlnu, view(lint, :, z)), wgt)
    end
    return lint, dnv
end
