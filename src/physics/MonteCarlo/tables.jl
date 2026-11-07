# Generation tables (gpumonty `weights.cu` `init_weight_table`, `kernels.cu` `init_zone`):
# the superphoton weight w(ν) = L_ν/(h N_s) on a log-frequency grid, and the per-zone
# target count N_s = ∫ dlnν (∫ j_ν dΩ) dV/(h w) with its rejection envelope
# max_ν (∫ j_ν dΩ)/w. The records and the per-zone kernel are here; the threaded
# builders are `Data.build_*`.
#
# The count is the trapezoid rule over the n + 1 weight nodes (half weight on the two end
# nodes), the quadrature of the ln ν integral that the generation density samples: with
# w_m = Σ_z ∫j dΩ(ν_m) dV_z/(h N_s') and N_s' = N_s/ln(ν_max/ν_min), the counts of all
# zones sum to N_s' dlnν Σ_m c_m = N_s exactly (c_m = 1/2 at the ends, 1 inside).

"Lower edge (Hz) of the default generation band (gpumonty sphere model `NUMIN`)."
const MC_NU_MIN = 1.0e8
"Upper edge (Hz) of the default generation band."
const MC_NU_MAX = 1.0e16
"Default number of log-frequency intervals of the weight table."
const MC_N_ESAMP = 400
"Floor added to a weight or an emissivity before dividing by it or taking its log, so a zone that does not emit never divides by zero."
const WEIGHT_FLOOR = 1.0e-100

"""
    WeightTable{V}(lnu_min, lnu_max, dlnu, lwgt)

The superphoton weight `w(ν)` (photons per packet): `lwgt[i] = log w` at
`ν_i = exp(lnu_min + (i − 1) dlnu)`, `i = 1:n`, over `[exp(lnu_min), exp(lnu_max)]`.
`isbits` after `Adapt.adapt` to a device.
"""
struct WeightTable{V<:AbstractVector{Float64}}
    lnu_min::Float64
    lnu_max::Float64
    dlnu::Float64
    lwgt::V
end
Adapt.@adapt_structure WeightTable

"""
    weight_at(nu, wgt::WeightTable) -> Float64

`w(ν)` by linear interpolation of `log w` in `ln ν` (gpumonty `linear_interp_weight`),
clamped to the first and last nodes.
"""
@inline weight_at(nu::Real, wgt::WeightTable) =
    loglinear_at(nu, wgt.lwgt, wgt.lnu_min, wgt.dlnu, length(wgt.lwgt))

"""
    node_weight(m, n) -> Float64

Trapezoid weight of node `m ∈ 0:n` of an `n`-interval rule: `1/2` at `m = 0` and
`m = n`, 1 inside.
"""
@inline node_weight(m::Integer, n::Integer) = (m == 0 || m == n) ? 0.5 : 1.0

"""
    mean_weight(wgt::WeightTable) -> Float64

The mean weight of the primaries, `Σ_m c_m w_m / n` over the `n + 1` nodes with the
trapezoid weights `c_m` ([`node_weight`](@ref)): the emitted photon rate over the target
count, `Ṅ/N_s`, since both are the same trapezoid sums.
"""
function mean_weight(wgt::WeightTable)
    n = length(wgt.lwgt) - 1
    n >= 1 || return exp(wgt.lwgt[1])
    acc = 0.0
    @inbounds for m in 0:n
        acc += node_weight(m, n) * exp(wgt.lwgt[m + 1])
    end
    return acc / n
end

"""
    zone_count(n_e, theta_e, B, dV, wgt::WeightTable, emis::EmissivityTable) -> (N_s, dnmax)

Expected superphoton count and frequency-rejection envelope of a zone of plasma
`(n_e, Θₑ, B)` and proper volume `dV` (cm³) on the nodes `ν_m` of `wgt`:
`N_s = (dV/h) dlnu Σ_m c_m ∫j_ν dΩ(ν_m)/(w_m + WEIGHT_FLOOR)` with the trapezoid
weights `c_m` and `dnmax = max_m ∫j_ν dΩ(ν_m)/(w_m + WEIGHT_FLOOR)`, `∫j_ν dΩ` from
[`int_jnu`](@ref) with `emis`.
"""
function zone_count(n_e, theta_e, B, dV, wgt::WeightTable, emis::EmissivityTable)
    n = length(wgt.lwgt) - 1
    dnmax = 0.0
    ninterp = 0.0
    @inbounds for m in 0:n
        nu = exp(m * wgt.dlnu + wgt.lnu_min)
        w = exp(wgt.lwgt[m + 1])
        dn = int_jnu(emis, n_e, theta_e, B, nu) / (w + WEIGHT_FLOOR)
        dnmax = max(dnmax, dn)
        ninterp += node_weight(m, n) * dn
    end
    return dV / HPL * (wgt.dlnu * ninterp), dnmax
end

"""
    zone_nsuperphotons(zone::EmissionZone, wgt::WeightTable[, emis = EMISSIVITY_TABLE])
        -> (N_s, dnmax)

[`zone_count`](@ref) of the plasma and volume of `zone` (gpumonty `init_zone`).
"""
zone_nsuperphotons(zone::EmissionZone, wgt::WeightTable, emis::EmissivityTable = EMISSIVITY_TABLE) =
    zone_count(zone.n_e, zone.theta_e, zone.B, zone.dV, wgt, emis)
