# Host builders of the Monte Carlo tables whose lookups are `MonteCarlo` physics: the
# hot-cross table (gpumonty `hotcross.dat` reader and the quadrature build), the
# superphoton weight table, the per-zone emissivity table with the per-zone counts, and
# the tabulated emissivities of tabulated generation (gpumonty `weights.cu`,
# `kernels.cu` `init_zone`). Each threaded loop writes disjoint columns or merges exact
# partial sums by integer addition, so every table is independent of the thread count.

using ..Numerics: HPL, xsum_zeros, xsum_add!, xsum_round, XSUM_MAX_ADDENDS
using ..Matter: EmissionZone
using ..MonteCarlo: HotcrossTable, NW, NT, MINW, MAXW, MINT, MAXT, total_compton_cross_num,
                    WeightTable, ZoneEmissionTable, IntJnuTable, int_jnu_zone, tabulated_dnmax,
                    WEIGHT_FLOOR, MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP, INT_JNU_REFINE

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
`(NW + 1) × (NT + 1)` log-log grid, threaded over the energy nodes.
"""
function build_hotcross_table()
    logsigma = Matrix{Float64}(undef, NW + 1, NT + 1)
    dlw = log10(MAXW / MINW) / NW
    dlT = log10(MAXT / MINT) / NT
    lminw = log10(MINW)
    lmint = log10(MINT)
    Threads.@threads for i in 0:NW
        lw = lminw + i * dlw
        @inbounds for j in 0:NT
            lT = lmint + j * dlT
            logsigma[i + 1, j + 1] = log10(total_compton_cross_num(10.0^lw, 10.0^lT))
        end
    end
    return HotcrossTable(logsigma)
end

"""
    build_weight_table(zones, Ns; n_esamp = MC_N_ESAMP, nu_min = MC_NU_MIN, nu_max = MC_NU_MAX)
        -> WeightTable

The weight table `w(ν_m) = Σ_z ∫j_ν dΩ(zone_z, ν_m) dV_z / (h Ns)` on `n_esamp + 1`
log-spaced nodes over `[nu_min, nu_max]`, stored as `log(w + WEIGHT_FLOOR)`; `Ns` is
the target number of superphotons (gpumonty `init_weight_table`). Serial sum over the
zones in list order.
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
    build_zone_emission_table(zones; n_esamp = MC_N_ESAMP, nu_min = MC_NU_MIN,
                              nu_max = MC_NU_MAX) -> ZoneEmissionTable

`∫j_ν dΩ` of every zone on the `n_esamp + 1` nodes of the weight grid over
`[nu_min, nu_max]`, threaded over zones.
"""
function build_zone_emission_table(zones::AbstractVector{<:EmissionZone};
                                   n_esamp::Integer = MC_N_ESAMP, nu_min::Real = MC_NU_MIN,
                                   nu_max::Real = MC_NU_MAX)
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
    build_weight_table(jt::ZoneEmissionTable, Ns; nthreads = Threads.nthreads()) -> WeightTable

The weight table of the zones of `jt`, `w(ν_m) = Σ_z jnu[m, z] dV[z] / (h Ns)`, with
each zone sum exact (integer limbs, rounded once), so the table is the correctly
rounded value whatever `nthreads`, the number of chunks of zones summed in parallel.
"""
function build_weight_table(jt::ZoneEmissionTable, Ns::Real;
                            nthreads::Integer = Threads.nthreads())
    n_nodes = jt.n_esamp + 1
    nz = size(jt.jnu, 2)
    hNs = HPL * Float64(Ns)
    nchunks = clamp(Int(nthreads), 1, max(1, nz))
    nz <= XSUM_MAX_ADDENDS || error("build_weight_table: too many zones for an exact sum")
    partials = [xsum_zeros(n_nodes) for _ in 1:nchunks]
    Threads.@threads for c in 1:nchunks
        acc = partials[c]
        lo = div((c - 1) * nz, nchunks) + 1
        hi = div(c * nz, nchunks)
        @inbounds for z in lo:hi
            dVz = jt.dV[z]
            for m in 1:n_nodes
                xsum_add!(acc, jt.jnu[m, z] * dVz, m)
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
    return WeightTable(jt.lnu_min, jt.lnu_max, jt.dlnu, lwgt)
end

"""
    zone_counts(jt::ZoneEmissionTable, wgt::WeightTable; nthreads = Threads.nthreads())
        -> (Ns::Vector{Float64}, dnmax::Vector{Float64})

Expected superphoton count and frequency-rejection envelope of every zone of `jt`, as
`MonteCarlo.zone_nsuperphotons` computes them, from the tabulated emissivities.
Threaded over zones. `jt` and `wgt` must share the frequency nodes.
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
            dn = jt.jnu[m, z] / (w + WEIGHT_FLOOR)
            dnmax = max(dnmax, dn)
            ninterp += wgt.dlnu * dn
        end
        @inbounds Nsv[z] = jt.dV[z] / HPL * ninterp
        @inbounds dnv[z] = dnmax
    end
    return Nsv, dnv
end

"""
    build_int_jnu_table(zone, wgt; refine = INT_JNU_REFINE) -> IntJnuTable

Tabulated `∫j_ν dΩ` of `zone` on the `refine`-fold refinement of the nodes of `wgt`
(every weight node is a table node). Interior nodes hold the smaller of the exact
logarithm and the chord between the neighbouring weight nodes, so the tabulated
acceptance clips where the exact one would exceed 1 and the envelope
(`tabulated_dnmax`) stays the maximum over the weight nodes.
"""
function build_int_jnu_table(zone::EmissionZone, wgt::WeightTable;
                             refine::Integer = INT_JNU_REFINE)
    k = Int(refine)
    n_coarse = length(wgt.lwgt) - 1
    n_fine = k * n_coarse
    dlnu = (wgt.lnu_max - wgt.lnu_min) / n_fine
    lc = Vector{Float64}(undef, n_coarse + 1)
    @inbounds for m in 0:n_coarse
        nu = exp(m * wgt.dlnu + wgt.lnu_min)
        lc[m + 1] = log(int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) + WEIGHT_FLOOR)
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
            lx = log(int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) + WEIGHT_FLOOR)
            ljnu[i + 1] = min(lx, chord)
        end
    end
    return IntJnuTable(wgt.lnu_min, dlnu, ljnu)
end

"""
    zone_int_jnu_tables(jt::ZoneEmissionTable, wgt::WeightTable) -> (lint, dnmax)

Tabulated generation for every zone of `jt` on the weight nodes (refinement 1):
`lint[m, z] = log(jnu[m, z] + WEIGHT_FLOOR)` and `dnmax[z]`, the envelope of
`tabulated_dnmax` over column `z`. Threaded over zones. `jt` and `wgt` must share the
frequency nodes.
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
            lint[m, z] = log(jt.jnu[m, z] + WEIGHT_FLOOR)
        end
        dnv[z] = tabulated_dnmax(IntJnuTable(jt.lnu_min, jt.dlnu, view(lint, :, z)), wgt)
    end
    return lint, dnv
end
