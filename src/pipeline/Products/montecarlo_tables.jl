# The tables of a Monte Carlo run on the run's executor: the emission zones of a plasma
# grid with their generation tables, and the process's hot-cross table. Threads build them
# on the host (`Data.zone_tables`, `Data.build_hotcross_table`); a pool builds them on its
# devices as programs with the same arithmetic. The zone tables are two passes: the
# per-node sums of ∫j_ν dΩ(ν_m) dV_z in exact integer limbs (each item one node over a
# chunk of zones, in a local accumulator merged by integer atomics), then every zone's
# count and envelope by `MonteCarlo.zone_count`; the hot-cross table is the cross section
# at every node, its logarithm taken on the host. Every operation of a per-zone or
# per-node value is an IEEE add, multiply, divide, square root or fma (the logarithms and
# exponentials are `tang_log` and `tang_exp`; the node coordinates and the Jüttner
# normalisations come from the host), so a device computes the host's bits, and exact
# sums do not depend on the grouping: the tables are bitwise those of the host on any
# executor. The zones and tables of the last grid run are kept for the next run on the
# same blocks and configuration.

using ..Spacetime: stop_bound

"""
    ZoneSums(sites, nus, emissivity, chunk, sums)

Pass 1 of the zone tables on an executor, called as `prog(item, i)` without outputs:
item `c n + m` (`n = length(nus)` nodes, `m ∈ 1:n`, zone chunk `c ≥ 0`) sums
`∫j_ν dΩ(nus[m]) dV_z` over the zones `z` of chunk `c` (`chunk` consecutive entries of
`sites`) exactly in its own limbs and adds them to scalar `m` of the `ExactScalars`
`sums` (`Execution.add_limbs!`). `∫j_ν dΩ` is `int_jnu` with the `EmissivityTable`
`emissivity`. `isbits` after `Adapt.adapt`.
"""
struct ZoneSums{Z,V,E,S} <: Function
    sites::Z
    nus::V
    emissivity::E
    chunk::Int
    sums::S
end
Adapt.@adapt_structure ZoneSums

function (p::ZoneSums)(item, i)
    nn = length(p.nus)
    c, m = divrem(item - 1, nn)
    lo = c * p.chunk + 1
    hi = min(lo + p.chunk - 1, length(p.sites))
    nu = @inbounds p.nus[m + 1]
    acc = zero(MVector{XSUM_NSLOT,Int64})
    @inbounds for z in lo:hi
        s = p.sites[z]
        xsum_add!(acc, int_jnu(p.emissivity, s.n_e, s.theta_e, s.B, nu) * s.dV)
    end
    Execution.add_limbs!(p.sums, m + 1, acc, hi - lo + 1)
    return nothing
end

"""
    ZoneCounts(sites, nodes, emissivity)

Pass 2 of the zone tables on an executor: item `z` writes the expected count and the
frequency envelope of zone `sites[z]`, `MonteCarlo.zone_count` on the `ZoneNodes`
`nodes` with `emissivity`, at `Nz[i]` and `dnmax[i]`. `isbits` after `Adapt.adapt`.
"""
struct ZoneCounts{Z,N,E} <: Function
    sites::Z
    nodes::N
    emissivity::E
end
Adapt.@adapt_structure ZoneCounts

function (p::ZoneCounts)(item, i, Nz, dnmax)
    s = @inbounds p.sites[item]
    a, b = zone_count(s.n_e, s.theta_e, s.B, s.dV, p.nodes, p.emissivity)
    @inbounds Nz[i] = a
    @inbounds dnmax[i] = b
    return nothing
end

# Zones per item of `ZoneSums` over `nz` zones: about 512 chunks per node, 256 to 8192
# zones each. The sums do not depend on it.
zone_chunk(nz::Integer) = clamp(cld(Int(nz), 512), 256, 8192)

"""
    grid_tables(exec, sites, Ns; n_esamp = MC_N_ESAMP, nu_min = MC_NU_MIN,
                nu_max = MC_NU_MAX, emissivity = EMISSIVITY_TABLE) -> (wgt, Nz, dnmax)

`Data.zone_tables(sites, Ns; …)` on the executor `exec`: on threads the host builder, on
a pool its two passes as the programs [`ZoneSums`](@ref) and [`ZoneCounts`](@ref) on the
pool's devices. Bitwise the same tables on every executor.
"""
grid_tables(::ThreadedExecutor, sites, Ns; kw...) = zone_tables(sites, Ns; kw...)

function grid_tables(exec::Execution.PoolExecutor, sites, Ns::Real;
                     n_esamp::Integer = MC_N_ESAMP, nu_min::Real = MC_NU_MIN,
                     nu_max::Real = MC_NU_MAX, emissivity = EMISSIVITY_TABLE)
    lnu_min = log(nu_min)
    lnu_max = log(nu_max)
    dlnu = (lnu_max - lnu_min) / n_esamp
    nn = Int(n_esamp) + 1
    nz = length(sites)
    hNs = HPL * Float64(Ns)
    nz <= XSUM_MAX_ADDENDS || error("grid_tables: too many zones for an exact sum")
    nus = [exp(m * dlnu + lnu_min) for m in 0:(nn - 1)]
    sums = ExactScalars(nn)
    chunk = zone_chunk(nz)
    foreach_item!(exec, ZoneSums(sites, nus, emissivity, chunk, sums), 1:(cld(nz, chunk) * nn))
    gather!(sums)
    wgt = WeightTable(lnu_min, lnu_max, dlnu,
                      [log(scalar(sums, m) / hNs + WEIGHT_FLOOR) for m in 1:nn])
    Nz = Vector{Float64}(undef, nz)
    dnmax = Vector{Float64}(undef, nz)
    foreach_item!(exec, ZoneCounts(sites, ZoneNodes(wgt), emissivity), 1:nz;
                  outs = (Nz, dnmax))
    return wgt, Nz, dnmax
end

# The zones of the last grid run (its blocks held weakly, through their data array) and
# the tables of its generation settings.
mutable struct ZoneMemo
    data::WeakRef
    blocks::Any
    zkey::Any
    zones::Any
    tkey::Any
    tables::Any
end
const ZONE_MEMO = ZoneMemo(WeakRef(nothing), nothing, nothing, nothing, nothing, nothing)
const ZONE_MEMO_LOCK = ReentrantLock()

"""
    check_stop_sphere(st, o, r_em)

Refuse a Monte Carlo run on `st` whose stop sphere leaves the sphere of
`Spacetime.stop_bound`, inside which `st` is its own metric. The march of the run
`o` stops a photon outbound past `max(o.march.R_em, r_em)` or past `o.march.escape`, one
step beyond at most, and a step advances the position by about `d/C` with `d` below one,
so the stop radius plus `1/o.march.C` must stay within the bound: on numerical-relativity
data the photons are then recorded in the metric of the slices.
"""
function check_stop_sphere(st, o, r_em)
    rho = stop_bound(st)
    rho == Inf && return nothing
    R = min(max(o.march.R_em, Float64(r_em)), o.march.escape)
    R + 1 / o.march.C <= rho ||
        throw(ArgumentError("mc_spectrum: the stop sphere of radius $R, with one step " *
                            "1/C = $(1 / o.march.C), leaves the lattice of " *
                            "$(nameof(typeof(st))), whose nearest face lies $rho from the " *
                            "origin; set R_em or escape, and r_emit_max (the zones reach " *
                            "1.2 times their radius), below $(rho - 1 / o.march.C)"))
    return nothing
end

"""
    grid_zones(blocks, st, params, exec, o) -> (sites, r_em, wgt, Nz, dnmax)

The emission zones of a grid run (`Data.emission_sites` of `blocks` on `st` with
`params` and the zone options of the run `o`), 1.2 times the largest radius of the escape
proxy over them, and their generation tables for `o.gen` on `exec` ([`grid_tables`](@ref)).
The last run's zones are reused by a run on the same `blocks` (the same object, not
changed in place since), spacetime, parameters and zone options (and seed, at a stride
above 1), and its tables by a run that also has the same generation settings; both are
what the run would build, bitwise. A run whose stop sphere leaves the lattice of an ADM
spacetime is refused ([`check_stop_sphere`](@ref)).
"""
function grid_zones(blocks::FluidBlocks, st, params, exec, o)
    zkey = (st, params, o.zones, o.zones.stride > 1 ? o.seed : nothing)
    g = o.gen
    tkey = (g.Ns, g.nu_min, g.nu_max, g.n_esamp)
    m = ZONE_MEMO
    hit = @lock ZONE_MEMO_LOCK (m.data.value === blocks.data && m.blocks === blocks &&
                                isequal(m.zkey, zkey))
    if hit
        zones = m.zones
    else
        sites = emission_sites(blocks, st, params; seed = o.seed, o.zones...)
        isempty(sites) &&
            throw(ArgumentError("mc_spectrum: no emitting zones (check the matter and the cuts)"))
        zones = (sites, maximum(z -> 1.2 * proxy_distance(st, z.x)[2], sites))
        @lock ZONE_MEMO_LOCK begin
            m.data = WeakRef(blocks.data); m.blocks = blocks; m.zkey = zkey; m.zones = zones
            m.tkey = nothing; m.tables = nothing
        end
    end
    check_stop_sphere(st, o, zones[2])
    tables = @lock ZONE_MEMO_LOCK (m.zones === zones && isequal(m.tkey, tkey)) ? m.tables : nothing
    if tables === nothing
        tables = grid_tables(exec, zones[1], g.Ns / (log(g.nu_max) - log(g.nu_min));
                             n_esamp = g.n_esamp, nu_min = g.nu_min, nu_max = g.nu_max)
        @lock ZONE_MEMO_LOCK if m.zones === zones
            m.tkey = tkey
            m.tables = tables
        end
    end
    return (zones..., tables...)
end

"""
    HotcrossSigma(eps, theta, k2f)

The hot cross section at the nodes of the hot-cross table on an executor: item
`k + 1 + n j` (`n = length(eps)`) writes `MonteCarlo.total_compton_cross_num(eps[k + 1],
theta[j + 1], k2f[j + 1])` at `out[i]`, with the nodes of `MonteCarlo.hotcross_nodes` and
the Jüttner normalisations `k2f` evaluated on the host. `isbits` after `Adapt.adapt`.
"""
struct HotcrossSigma{V} <: Function
    eps::V
    theta::V
    k2f::V
end
Adapt.@adapt_structure HotcrossSigma

function (p::HotcrossSigma)(item, i, out)
    j, k = divrem(item - 1, length(p.eps))
    @inbounds out[i] = total_compton_cross_num(p.eps[k + 1], p.theta[j + 1], p.k2f[j + 1])
    return nothing
end

"""
    hotcross_build(exec) -> HotcrossTable

`Data.build_hotcross_table()` on the executor `exec`: on threads the host builder, on a
pool the cross sections of [`HotcrossSigma`](@ref) on its devices and their `log10` on the
host. Bitwise the same table on every executor.
"""
hotcross_build(::ThreadedExecutor) = build_hotcross_table()

function hotcross_build(exec::Execution.PoolExecutor)
    eps, theta = hotcross_nodes()
    sigma = Vector{Float64}(undef, length(eps) * length(theta))
    foreach_item!(exec, HotcrossSigma(eps, theta, juttner_norm.(theta)), 1:length(sigma);
                  outs = (sigma,))
    return HotcrossTable(reshape(log10.(sigma), length(eps), length(theta)))
end
