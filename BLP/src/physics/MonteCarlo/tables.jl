# Generation tables (gpumonty `weights.cu` `init_weight_table`, `kernels.cu` `init_zone`):
# the superphoton weight w(ν) = L_ν/(h N_s) on a log-frequency grid, the per-zone target
# count N_s = ∫ dlnν (∫ j_ν dΩ) dV/(h w) with its rejection envelope max_ν (∫ j_ν dΩ)/w,
# and the optional tabulated log ∫ j_ν dΩ of tabulated generation. The records and their
# lookups are here; the threaded builders are `Data.build_*`.

"Lower edge (Hz) of the default generation band (gpumonty sphere model `NUMIN`)."
const MC_NU_MIN = 1.0e8
"Upper edge (Hz) of the default generation band."
const MC_NU_MAX = 1.0e16
"Default number of log-frequency intervals of the weight table."
const MC_N_ESAMP = 400
"Floor added to a weight or an emissivity before dividing by it or taking its log, so a zone that does not emit never divides by zero."
const WEIGHT_FLOOR = 1.0e-100
"Default refinement of the tabulated-emissivity grid over the weight grid."
const INT_JNU_REFINE = 10

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
    ZoneEmissionTable{M,V}(lnu_min, lnu_max, dlnu, n_esamp, jnu, dV)

`∫ j_ν dΩ` of every zone of a list on the nodes of a weight grid:
`jnu[m + 1, z] = int_jnu_zone(zone_z, ν_m)`, `ν_m = exp(lnu_min + m dlnu)`,
`m = 0:n_esamp`, and `dV[z]` the proper volume (cm³) of zone `z`. Both the weight
table and the per-zone counts of a zone list are read from it.
"""
struct ZoneEmissionTable{M<:AbstractMatrix{Float64},V<:AbstractVector{Float64}}
    lnu_min::Float64
    lnu_max::Float64
    dlnu::Float64
    n_esamp::Int
    jnu::M
    dV::V
end

"""
    IntJnuTable{V}(lnu_min, dlnu, ljnu)

Tabulated `∫ j_ν dΩ` of one zone: `ljnu[i] = log(int_jnu_zone(ν_i) + WEIGHT_FLOOR)` at
`ν_i = exp(lnu_min + (i − 1) dlnu)`. `ljnu` may be a column view of a per-zone matrix.
`isbits` after `Adapt.adapt` to a device.
"""
struct IntJnuTable{V<:AbstractVector{Float64}}
    lnu_min::Float64
    dlnu::Float64
    ljnu::V
end
Adapt.@adapt_structure IntJnuTable

"""
    int_jnu_at(nu, gt::IntJnuTable) -> Float64

`∫ j_ν dΩ` at `nu` by linear interpolation of the logarithm in `ln ν`, clamped to the
first and last nodes.
"""
@inline int_jnu_at(nu::Real, gt::IntJnuTable) =
    loglinear_at(nu, gt.ljnu, gt.lnu_min, gt.dlnu, length(gt.ljnu))

"""
    zone_nsuperphotons(zone::EmissionZone, wgt::WeightTable) -> (N_s, dnmax)

Expected superphoton count and frequency-rejection envelope of `zone` on the nodes of
`wgt`: `N_s = (dV/h) Σ_m dlnu ∫j_ν dΩ/(w_m + WEIGHT_FLOOR)` and
`dnmax = max_m ∫j_ν dΩ/(w_m + WEIGHT_FLOOR)` (gpumonty `init_zone`).
"""
function zone_nsuperphotons(zone::EmissionZone, wgt::WeightTable)
    n_nodes = length(wgt.lwgt)
    dnmax = 0.0
    ninterp = 0.0
    @inbounds for m in 0:(n_nodes - 1)
        nu = exp(m * wgt.dlnu + wgt.lnu_min)
        w = exp(wgt.lwgt[m + 1])
        dn = int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) / (w + WEIGHT_FLOOR)
        dnmax = max(dnmax, dn)
        ninterp += wgt.dlnu * dn
    end
    N_s = zone.dV / HPL * ninterp
    return N_s, dnmax
end

"""
    tabulated_dnmax(gt::IntJnuTable, wgt::WeightTable) -> Float64

Rejection envelope of tabulated generation: the maximum over the nodes of `gt` of
`int_jnu_at / (weight_at + WEIGHT_FLOOR)`. Both logarithms are piecewise linear in
`ln ν` between these nodes when every weight node is a node of `gt`, so the maximum
over the nodes bounds the ratio at every frequency and the acceptance is at most 1.
"""
function tabulated_dnmax(gt::IntJnuTable, wgt::WeightTable)
    dnmax = 0.0
    @inbounds for i in eachindex(gt.ljnu)
        nu = exp((i - 1) * gt.dlnu + gt.lnu_min)
        dnmax = max(dnmax, exp(gt.ljnu[i]) / (weight_at(nu, wgt) + WEIGHT_FLOOR))
    end
    return dnmax
end
