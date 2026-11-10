# Host builders of the Monte Carlo tables whose lookups are `MonteCarlo` physics: the
# hot-cross table (gpumonty `hotcross.dat` reader and the quadrature build), the
# emission zones of a block mesh as compact records, the superphoton weight table and the
# per-zone counts (gpumonty `weights.cu`, `kernels.cu` `init_zone`). Each threaded loop
# writes disjoint slots or merges exact partial sums by integer addition, so every table
# is independent of the thread count.

using LinearAlgebra: det
using ..Numerics: HPL, xsum_zeros, xsum_add!, xsum_round, XSUM_MAX_ADDENDS, run_rng, mc_uniform,
                  MCRng
using ..Spacetime: proxy_distance, HORIZON_EPS_SAMPLED
using ..Matter: EmissionZone, ZoneSite, sample
using ..MonteCarlo: HotcrossTable, NW, NT, total_compton_cross_num, hotcross_nodes,
                    WeightTable, EmissivityTable, EMISSIVITY_TABLE, int_jnu, int_jnu_zone,
                    ZoneNodes, zone_count, WEIGHT_FLOOR, MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP

"""
    load_hotcross(path) -> HotcrossTable

Read a gpumonty `hotcross.dat` (one node per line: `i j log₁₀ε log₁₀Θₑ log₁₀σ_h`,
`(NW + 1)(NT + 1)` lines). Only the grid indices and `log₁₀ σ_h` are read, so the
table does not depend on the row order.
"""
function load_hotcross(path::AbstractString)
    logsigma = Matrix{Float64}(undef, NW + 1, NT + 1)
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            logsigma[i + 1, j + 1] = parse(Float64, parts[5])
        end
    end
    return HotcrossTable(logsigma)
end

"""
    build_hotcross_table() -> HotcrossTable

The hot-cross table from `total_compton_cross_num` at every node of the
`(NW + 1) × (NT + 1)` log-log grid (`MonteCarlo.hotcross_nodes`), threaded over the
energy nodes.
"""
function build_hotcross_table()
    logsigma = Matrix{Float64}(undef, NW + 1, NT + 1)
    eps, theta = hotcross_nodes()
    Threads.@threads for i in 0:NW
        @inbounds for j in 0:NT
            logsigma[i + 1, j + 1] = log10(total_compton_cross_num(eps[i + 1], theta[j + 1]))
        end
    end
    return HotcrossTable(logsigma)
end

"""
    build_weight_table(zones, Ns; n_esamp = MC_N_ESAMP, nu_min = MC_NU_MIN, nu_max = MC_NU_MAX)
        -> WeightTable

The weight table `w(ν_m) = Σ_z ∫j_ν dΩ(zone_z, ν_m) dV_z / (h Ns)` of a list of
`EmissionZone`s on `n_esamp + 1` log-spaced nodes over `[nu_min, nu_max]`, stored as
`log(w + WEIGHT_FLOOR)`; `Ns` is the target number of superphotons per unit `ln ν`
(gpumonty `init_weight_table`). Serial sum over the zones in list order.
"""
function build_weight_table(zones::AbstractVector{<:EmissionZone}, Ns::Real;
                            n_esamp::Integer = MC_N_ESAMP, nu_min::Real = MC_NU_MIN,
                            nu_max::Real = MC_NU_MAX)
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
        lwgt[m + 1] = log(s / hNs + WEIGHT_FLOOR)
    end
    return WeightTable(lnu_min, lnu_max, dlnu, lwgt)
end

"""
    emission_sites(blocks::FluidBlocks, st, params; stride = 1, seed = 0, r_emit_max = Inf,
                   max_zones = typemax(Int), t0 = 0.0) -> Vector{ZoneSite{Float64}}

The emission zones of the cell centres of `blocks`, at coordinate time `t0` (the time at
which `st` is read; the time of the snapshot on a time-dependent spacetime), as
[`ZoneSite`](@ref)s. Each block is
cut into groups of `stride` cells per axis (the last group of an axis takes the cells
left); the cells of a group that their own block owns (`locate`) are its `n_own` owned
cells, and one of them is drawn uniformly, by the uniform of draw `c` of the stream
`run_rng(seed, 0)`, `c` the group's global index (`stride = 1`: the cell itself). The
drawn cell is a zone when it lies above the horizon proxy
(`proxy_distance(st, x)[1] ≥ HORIZON_EPS_SAMPLED`: a photon emitted below is captured
before its first step), within the escape radius `r_emit_max`, and its plasma decoded on
`st` with `params` is valid with `n_e > 0` and `Θₑ > 0`. Its volume is
`n_own √−g Δx¹Δx²Δx³ L_unit³` at the drawn cell (`√−g` taken as 1 where `det g ≥ 0`), so
for any function `f` of the cell, `Σ_zones f dV` is an unbiased estimate of the sum of
`f √−g Δ³x L³` over every owned cell that passes the cuts, exact at `stride = 1`.
Threaded over blocks; the list is in block order, groups `k`, `j`, `i` outermost to
innermost, at most `max_zones` long, independent of the thread count.
"""
function emission_sites(blocks::FluidBlocks, st, params; stride::Integer = 1,
                        seed::Integer = 0, r_emit_max::Real = Inf,
                        max_zones::Integer = typemax(Int), t0::Real = 0.0)
    s = Int(stride)
    s >= 1 || throw(ArgumentError("emission_sites: stride must be positive"))
    ng = (cld(blocks.ni, s), cld(blocks.nj, s), cld(blocks.nk, s))
    ngroups = ng[1] * ng[2] * ng[3]
    blocks.nmb * ngroups <= typemax(UInt32) ||
        throw(ArgumentError("emission_sites: more than 2^32 groups"))
    rng0 = run_rng(seed, 0)
    Lu3 = params.units.L_unit^3
    rmax = Float64(r_emit_max)
    tz = Float64(t0)
    per_block = Vector{Vector{ZoneSite{Float64}}}(undef, blocks.nmb)
    Threads.@threads for mb in 1:blocks.nmb
        out = ZoneSite{Float64}[]
        d1 = blocks.dx[1, mb]; d2 = blocks.dx[2, mb]; d3 = blocks.dx[3, mb]
        cell(i, j, k) = SVector(tz, blocks.x_left[1, mb] + (i - 1) * d1,
                                blocks.x_left[2, mb] + (j - 1) * d2,
                                blocks.x_left[3, mb] + (k - 1) * d3)
        owned(x) = locate(blocks, x[2], x[3], x[4]) == mb
        for gk in 0:(ng[3] - 1), gj in 0:(ng[2] - 1), gi in 0:(ng[1] - 1)
            ks = (gk * s + 1):min((gk + 1) * s, blocks.nk)
            js = (gj * s + 1):min((gj + 1) * s, blocks.nj)
            is = (gi * s + 1):min((gi + 1) * s, blocks.ni)
            n_own = 0
            for k in ks, j in js, i in is
                owned(cell(i, j, k)) && (n_own += 1)
            end
            n_own == 0 && continue
            pick = 0
            if n_own > 1
                c = (mb - 1) * ngroups + (gk * ng[2] + gj) * ng[1] + gi
                u, _ = mc_uniform(MCRng(rng0.key, UInt32(c), rng0.sid))
                pick = min(unsafe_trunc(Int, u * n_own), n_own - 1)
            end
            xc = cell(first(is), first(js), first(ks))
            for k in ks, j in js, i in is
                x = cell(i, j, k)
                owned(x) || continue
                pick == 0 && (xc = x; break)
                pick -= 1
            end
            dist, r = proxy_distance(st, xc)
            (dist >= HORIZON_EPS_SAMPLED && r <= rmax) || continue
            state, g = sample(blocks, st, xc, params)
            (state.valid && state.n_e > 0.0 && state.theta_e > 0.0) || continue
            detg = det(g[1])
            dV = (detg < 0.0 ? sqrt(-detg) : 1.0) * (d1 * d2 * d3 * n_own) * Lu3
            push!(out, ZoneSite{Float64}(xc, dV, state.n_e, state.theta_e, state.B))
        end
        per_block[mb] = out
    end
    n = min(sum(length, per_block; init = 0), max(0, Int(max_zones)))
    sites = Vector{ZoneSite{Float64}}(undef, n)
    k = 0
    for v in per_block
        m = min(length(v), n - k)
        m > 0 && copyto!(sites, k + 1, v, 1, m)
        k += m
    end
    return sites
end

"""
    zone_tables(sites, Ns; n_esamp = MC_N_ESAMP, nu_min = MC_NU_MIN, nu_max = MC_NU_MAX,
                emissivity = EMISSIVITY_TABLE, nthreads = Threads.nthreads())
        -> (wgt::WeightTable, Nz::Vector{Float64}, dnmax::Vector{Float64})

The generation tables of the zones `sites` (any vector of records with `n_e`,
`theta_e`, `B`, `dV`: [`ZoneSite`](@ref)s or `EmissionZone`s) in two streaming passes
over the zones, threaded over `nthreads` chunks: (1) the weight table
`w(ν_m) = Σ_z ∫j_ν dΩ(ν_m) dV_z/(h Ns)` on the `n_esamp + 1` nodes, each node's sum exact
(integer limbs, rounded once), so the table is the correctly rounded value whatever
`nthreads`; (2) every zone's count and envelope, `MonteCarlo.zone_count` on that table
(trapezoid in `ln ν`, so `Σ_z Nz = Ns ln(nu_max/nu_min)` when every node emits). `Ns` is
the target number per unit `ln ν`. `∫j_ν dΩ` ([`int_jnu`](@ref) with `emissivity`) is
evaluated in both passes and never stored, so the memory is that of the zones and the
nodes.
"""
function zone_tables(sites::AbstractVector, Ns::Real; n_esamp::Integer = MC_N_ESAMP,
                     nu_min::Real = MC_NU_MIN, nu_max::Real = MC_NU_MAX,
                     emissivity::EmissivityTable = EMISSIVITY_TABLE,
                     nthreads::Integer = Threads.nthreads())
    lnu_min = log(nu_min)
    lnu_max = log(nu_max)
    dlnu = (lnu_max - lnu_min) / n_esamp
    n_nodes = Int(n_esamp) + 1
    nz = length(sites)
    hNs = HPL * Float64(Ns)
    nchunks = clamp(Int(nthreads), 1, max(1, nz))
    nz <= XSUM_MAX_ADDENDS || error("zone_tables: too many zones for an exact sum")
    nus = [exp(m * dlnu + lnu_min) for m in 0:(n_nodes - 1)]
    partials = [xsum_zeros(n_nodes) for _ in 1:nchunks]
    Threads.@threads for c in 1:nchunks
        acc = partials[c]
        @inbounds for z in (div((c - 1) * nz, nchunks) + 1):div(c * nz, nchunks)
            zs = sites[z]
            for m in 1:n_nodes
                xsum_add!(acc, int_jnu(emissivity, zs.n_e, zs.theta_e, zs.B, nus[m]) * zs.dV, m)
            end
        end
    end
    total = partials[1]
    for c in 2:nchunks
        total .+= partials[c]
    end
    lwgt = Vector{Float64}(undef, n_nodes)
    @inbounds for m in 1:n_nodes
        lwgt[m] = log(xsum_round(total, m) / hNs + WEIGHT_FLOOR)
    end
    wgt = WeightTable(lnu_min, lnu_max, dlnu, lwgt)
    nodes = ZoneNodes(wgt)
    Nz = Vector{Float64}(undef, nz)
    dnmax = Vector{Float64}(undef, nz)
    Threads.@threads for c in 1:nchunks
        @inbounds for z in (div((c - 1) * nz, nchunks) + 1):div(c * nz, nchunks)
            zs = sites[z]
            Nz[z], dnmax[z] = zone_count(zs.n_e, zs.theta_e, zs.B, zs.dV, nodes, emissivity)
        end
    end
    return wgt, Nz, dnmax
end
