# Monte Carlo spectra: superphotons generated from the plan's matter (the uniform sphere,
# or the emitting cells of a plasma grid), tracked through absorption and Compton
# scattering by `MonteCarlo.track!`, and binned exactly. A primary owns the stream of its
# global index and a child the stream of its genealogy; the bins, the ledger and the
# children are integer sums, so a spectrum is bitwise independent of the executor, the
# device count, the chunking and the order of arrival. Scattering runs as a cascade
# (`Execution.foreach_cascade!`): within each chunk of primaries, layer n tracks the
# children that layer n − 1 spawned.
#
# Normalisation: the weight table fixes w(ν) = Σ_zones ∫j_ν dΩ dV/(h Ns/ln(ν_max/ν_min)),
# so the generated frequency density is uniform in ln ν, about `Ns` packets are launched,
# and ν L_ν(bin) = (mₑc²/dlnν) Σ w E estimates the emitted spectrum with no free constant.

"""
    ScatterResult

The spectrum of the scattering sphere ([`mc_spectrum`](@ref) with `scattering = true`).
`orders[n + 1]` is the escaped `Spectrum` of the photons of scattering order `n`
(`orders[end]` holds the order `max_scatt`), `total` the exact sum of the orders.
Diagnostics: `n_primary` primaries and `n_children` children tracked, the weight of the
primaries `w_primary`, the weight spawned into children `w_children`, of which
`w_first_order` by primaries, and `scatter_loss`, the incident weight lost to scattering
(`w_children ≈ scatter_loss` checks the unbiasedness of the exponential transform). `plan`
and `run` as in [`Image`](@ref).
"""
struct ScatterResult{S,P,R}
    orders::Vector{S}
    total::S
    max_scatt::Int
    n_primary::Int
    n_children::Int
    w_primary::Float64
    w_children::Float64
    w_first_order::Float64
    scatter_loss::Float64
    plan::P
    run::R
end

Base.show(io::IO, r::ScatterResult) =
    print(io, "ScatterResult(", length(r.orders), " orders, ", r.n_primary, " primaries, ",
          r.n_children, " children, ", r.run.executor.kind, ")")

"""
    GRMHDSpectrumResult

The spectrum of a plasma grid on a spacetime ([`mc_spectrum`](@ref)). `spectra[k]` is
the one-row escaped `Spectrum` of the photons whose asymptotic polar angle falls in the
inclination band `incl_deg[k] ± half_width_deg` (degrees), of solid angle `dOmega[k]`
(`nuLnu_solid(spectra[k], dOmega[k])`, or [`incl_nuLnu`](@ref), gives the
isotropic-equivalent `ν L_ν`); `total` is the all-sky spectrum. Diagnostics:
`emitted_power` and `escaped_power` (`mₑc² Σ w E` over the primaries at emission and over
the escaped photons, erg s⁻¹), `n_zones` emitting zones, `n_gen` primaries, `n_recorded`
escaped photons, and `scattering`, whether Compton scattering was tracked. `plan` and
`run` as in [`Image`](@ref).
"""
struct GRMHDSpectrumResult{S,P,R}
    incl_deg::Vector{Float64}
    half_width_deg::Float64
    dOmega::Vector{Float64}
    spectra::Vector{S}
    total::S
    emitted_power::Float64
    escaped_power::Float64
    n_zones::Int
    n_gen::Int
    n_recorded::Int
    scattering::Bool
    plan::P
    run::R
end

Base.show(io::IO, r::GRMHDSpectrumResult) =
    print(io, "GRMHDSpectrumResult(", length(r.spectra), " bands, ", r.n_zones, " zones, ",
          r.n_gen, " primaries, ", r.n_recorded, " escaped, ", r.run.executor.kind, ")")

"""
    incl_nuLnu(res::GRMHDSpectrumResult) -> (nu, L)

The isotropic-equivalent `ν L_ν` (erg s⁻¹) of every inclination band: `nu` the bin
centres, `L[k, :]` the spectrum of band `k`, `nuLnu_solid(res.spectra[k], res.dOmega[k])`.
"""
function incl_nuLnu(res::GRMHDSpectrumResult)
    nu = nu_centers(res.total)
    L = zeros(Float64, length(res.spectra), length(nu))
    for k in eachindex(res.spectra)
        L[k, :] .= nuLnu_solid(res.spectra[k], res.dOmega[k])[2]
    end
    return nu, L
end

"""
    analytic_nuLnu(sphere::UniformSphere, nu) -> Float64
    analytic_nuLnu(plan, nu) -> Float64

The optically thin spectrum of the uniform sphere, `ν L_ν = ν V ∫ j_ν dΩ` (erg s⁻¹) with
`V` the volume of the ball: the angle-integrated convention of `nuLnu_total`, which the
thin-sphere Monte Carlo estimates. The plan form takes the sphere of a plan of
`prepare(sphere)`.
"""
analytic_nuLnu(s::UniformSphere, nu::Real) = nu * s.dV * int_jnu_zone(s.n_e, s.theta_e, s.B, nu)
analytic_nuLnu(plan::Plan, nu::Real) = analytic_nuLnu(sphere_of(plan), nu)

function sphere_of(plan::Plan)
    plan.provider isa UniformSphere ||
        throw(ArgumentError("not a sphere plan: the matter is " *
                            "$(nameof(typeof(plan.provider))); prepare it with prepare(UniformSphere(…))"))
    return plan.provider
end

# ---------------------------------------------------------------- emission zones

"""
    emission_zones(plan; stride = 1, r_emit_max = Inf, max_zones = typemax(Int))
        -> Vector{EmissionZone{Float64}}
    emission_zones(blocks::FluidBlocks, st, params; stride, r_emit_max, max_zones)

One emission zone per cell centre of the plan's matter (an `AthenaKSnapshot` or
`FluidBlocks`; every `stride`-th cell per axis) that its own block owns (`locate`, so a
cell under a finer block is skipped), within Kerr–Schild radius `r_emit_max`, whose
plasma decoded on the plan's spacetime with the plan's parameter set is valid with
`n_e > 0` and `Θₑ > 0`. The zone's volume is the proper volume
`√−g Δx¹Δx²Δx³ stride³ L_unit³` (cm³), `√−g` taken as 1 where `det g ≥ 0`. Cells are
visited block by block, `k`, `j`, `i` outermost to innermost; at most `max_zones` zones.
"""
emission_zones(plan::Plan; kw...) =
    emission_zones(zone_blocks(plan.provider), plan.spacetime, plan.params; kw...)

function emission_zones(blocks::FluidBlocks, st, params; stride::Integer = 1,
                        r_emit_max::Real = Inf, max_zones::Integer = typemax(Int))
    s = Int(stride)
    s >= 1 || throw(ArgumentError("emission_zones: stride must be positive"))
    zones = EmissionZone{Float64}[]
    Lu3 = params.units.L_unit^3
    rmax = Float64(r_emit_max)
    for mb in 1:blocks.nmb
        d1 = blocks.dx[1, mb]; d2 = blocks.dx[2, mb]; d3 = blocks.dx[3, mb]
        dVcode = d1 * d2 * d3 * Float64(s)^3
        for k in 1:s:blocks.nk, j in 1:s:blocks.nj, i in 1:s:blocks.ni
            xc = SVector(0.0, blocks.x_left[1, mb] + (i - 1) * d1,
                         blocks.x_left[2, mb] + (j - 1) * d2, blocks.x_left[3, mb] + (k - 1) * d3)
            locate(blocks, xc[2], xc[3], xc[4]) == mb || continue
            proxy_distance(st, xc)[2] <= rmax || continue
            state, g = sample(blocks, st, xc, params)
            (state.valid && state.n_e > 0.0 && state.theta_e > 0.0) || continue
            detg = det(g[1])
            push!(zones, EmissionZone(state, (detg < 0.0 ? sqrt(-detg) : 1.0) * dVcode * Lu3, xc, g))
            length(zones) >= max_zones && return zones
        end
    end
    return zones
end

zone_blocks(b::FluidBlocks) = b
zone_blocks(s::AthenaKSnapshot) = s.blocks
zone_blocks(p) =
    throw(ArgumentError("emission zones need an AthenaKSnapshot or FluidBlocks as matter; " *
                        "got $(nameof(typeof(p)))"))

# ---------------------------------------------------------------- the product

"""
    mc_spectrum(plan, Ns; scattering = false, backend = nothing, bands = nothing,
                seed = 0x5eed5eed, nu_min = 1e8, nu_max = 1e16, spec_nu_min = nu_min,
                spec_nu_max = scattering ? 1e24 : nu_max, n_esamp = 400, n_ebins = 200,
                tabulated = false, table = nothing, max_scatt = 3, bias = nothing,
                absorption, capacity = nothing, C = plan.budget.C, escape = 1e3,
                R_em = 100.0, stride = 1, r_emit_max = Inf, max_zones = typemax(Int))
        -> Spectrum | ScatterResult | GRMHDSpectrumResult

The Monte Carlo spectrum of the plan's matter from about `Ns` superphotons, run on the
executor `resolve(backend)` (`nothing`: the session default).

The matter decides the run. The uniform sphere of `prepare(sphere)` (flat spacetime):
primaries uniform in the ball, a straight-line march (`FlatMarch`); without scattering
the result is the absorbed `Spectrum`, with scattering a [`ScatterResult`](@ref) of
per-order spectra (`absorption` defaults to `false` there: the idealized Comptonization
sphere). A plasma grid (`AthenaKSnapshot`, `FluidBlocks`) on a stationary spacetime:
primaries spread over the [`emission_zones`](@ref) of the plan (`stride`, `r_emit_max`,
`max_zones`) by their expected counts, rounded stochastically on the stream
`MCRng(seed, 0)`; a geodesic march (`CurvedMarch`: step divisor `C`, escape at radius
`escape`, emission-confinement radius the larger of `R_em` and 1.2 times the outermost
zone radius); absorption always by default; the result a
[`GRMHDSpectrumResult`](@ref).

Generation: the weight table over `[nu_min, nu_max]` on `n_esamp` intervals for about
`Ns` primaries, exact (`tabulated = false`) or tabulated frequency rejection. Primary `g`
draws the stream `primary_rng(seed, g)`. Transport: absorption (`absorption`), Compton
scattering (`scattering`) with the hot-cross `table` (`nothing`: `hotcross_table(plan)`,
built at the plan's first scattering run and kept in the plan), at most `max_scatt`
scatterings per photon, the importance `bias` (a `Bias`, its `b0`, or `nothing`: the
sphere's Thomson depth, 1 on a grid), lengths in `plan.params.units.L_unit` cm (one cm
for a plan without parameters). Binning:
`bands` (`PolarBands(6)` for the sphere, `InclinationBands((17, 50, 90), 10)` on a grid,
which takes inclination bands only), `n_ebins` log-frequency bins over
`[spec_nu_min, spec_nu_max]`.

Scattering runs as a cascade over chunks of primaries ([`Execution.foreach_cascade!`](@ref)):
within a chunk, each layer's children are tracked by the next layer, up to `max_scatt`
layers, and stay on the worker that runs the chunk. On devices a layer starts with a
child buffer of `capacity` slots (`nothing`: at least 4096, half the chunk's primaries);
a layer whose buffer overflows is rerun into zeroed scratch accumulators with a buffer of
the counted size, so no child is lost or counted twice. On threads each task's two child
buffers start with `capacity` slots (`nothing`: 4096) and grow; a photon whose children
overflow is rerun the same way.

Determinism: bitwise reduction. The bins, the ledger and the children depend only on the
plan, `Ns` and the keywords, whatever the executor, the device count, the chunking or
the buffer capacity.
"""
function mc_spectrum(plan::Plan, Ns::Real; scattering::Bool = false, backend = nothing,
                     bands = nothing, seed::Integer = 0x5eed5eed, nu_min::Real = MC_NU_MIN,
                     nu_max::Real = MC_NU_MAX, spec_nu_min::Real = nu_min,
                     spec_nu_max::Real = scattering ? 1.0e24 : nu_max,
                     n_esamp::Integer = MC_N_ESAMP, n_ebins::Integer = 200,
                     tabulated::Bool = false, table = nothing, max_scatt::Integer = 3,
                     bias = nothing,
                     absorption::Bool = !(scattering && plan.provider isa UniformSphere),
                     capacity = nothing, C::Real = plan.budget.C, escape::Real = 1.0e3,
                     R_em::Real = 100.0, stride::Integer = 1, r_emit_max::Real = Inf,
                     max_zones::Integer = typemax(Int))
    t0 = time_ns()
    exec = resolve(backend)
    Ns > 0 || throw(ArgumentError("mc_spectrum: Ns must be positive"))
    K = scattering ? Int(max_scatt) : 0
    K >= 0 || throw(ArgumentError("mc_spectrum: max_scatt must be non-negative"))
    capacity === nothing || capacity >= 1 ||
        throw(ArgumentError("mc_spectrum: capacity must be positive"))
    o = (seed = UInt32(seed & 0xffffffff), scattering = scattering, absorption = absorption,
         K = K,
         gen = (nu_min = Float64(nu_min), nu_max = Float64(nu_max), n_esamp = Int(n_esamp),
                tabulated = tabulated, Ns = Float64(Ns)),
         grid = (nu_min = Float64(spec_nu_min), nu_max = Float64(spec_nu_max),
                 n_ebins = Int(n_ebins)),
         table = scattering ? hotcross(table, plan) : table, bias = bias, capacity = capacity,
         march = (C = Float64(C), escape = Float64(escape), R_em = Float64(R_em)),
         zones = (stride = Int(stride), r_emit_max = Float64(r_emit_max),
                  max_zones = Int(max_zones)))
    return mc_run(plan.provider, plan, exec, bands, o, t0)
end

# The hot-cross table of a run: the one given, or the plan's.
hotcross(::Nothing, plan) = hotcross_table(plan)
hotcross(table, plan) = table

# The importance bias of a run.
mc_bias(::Nothing, s::UniformSphere) = Bias(s)
mc_bias(::Nothing, provider) = Bias(1.0)
mc_bias(b::Real, provider) = Bias(Float64(b))
mc_bias(b::Bias, provider) = b

# The opacities of a run: lengths in the plan's units, or in cm without a parameter set.
mc_opacities(::Nothing, provider, o) =
    Opacities(; table = o.table, bias = mc_bias(o.bias, provider), max_scatt = o.K)
mc_opacities(params, provider, o) =
    Opacities(params; table = o.table, bias = mc_bias(o.bias, provider), max_scatt = o.K)

mc_run(::Nothing, plan, exec, bands, o, t0) =
    throw(ArgumentError("mc_spectrum needs matter: prepare the plan with `matter = …`"))

# The uniform sphere.
function mc_run(sphere::UniformSphere, plan, exec, bands, o, t0)
    plan.spacetime isa Minkowski ||
        throw(ArgumentError("the Monte Carlo sphere runs on Minkowski(); the plan's " *
                            "spacetime is $(nameof(typeof(plan.spacetime)))"))
    tables, N = sphere_tables(sphere, o.gen)
    ctl = FlatMarch(sphere)
    transport = MCTransport(ctl, plan.spacetime, sphere, mc_opacities(plan.params, sphere, o),
                            Val(o.absorption), Val(o.scattering))
    rows = bands === nothing ? PolarBands(6) : bands
    orders = ntuple(_ -> ExactBins(rows; o.grid...), o.K + 1)
    ledger = mc_ledger()
    layout = MCStreamLayout(N, SPHERE_MAX_STEPS, o.K)
    nchildren = run_layers(exec, SphereSource(sphere, tables), transport, o.seed, N, orders,
                           ledger, layout, o.scattering, o.capacity)
    rec = run_record(exec, plan, t0)
    o.scattering || return Spectrum(orders[1]; plan = plan, run = rec)
    total = similar(orders[1])
    foreach(b -> merge!(total, b), orders)
    return ScatterResult([Spectrum(b) for b in orders], Spectrum(total), o.K, N, nchildren,
                         ledger_sum(ledger, PRIMARY_WEIGHT), ledger_sum(ledger, SPAWNED),
                         ledger_sum(ledger, SPAWNED_BY_PRIMARIES),
                         ledger_sum(ledger, SCATTER_LOSS), plan, rec)
end

# The one-zone generation tables of the sphere for about `Ns` primaries over the band,
# and the number of primaries.
function sphere_tables(sphere::UniformSphere, g)
    zone = table_zone(sphere)
    wgt = build_weight_table([zone], g.Ns / (log(g.nu_max) - log(g.nu_min));
                             n_esamp = g.n_esamp, nu_min = g.nu_min, nu_max = g.nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    N = max(1, round(Int, N_target))
    g.tabulated || return GenerationTables(wgt, dnmax), N
    gt = build_int_jnu_table(zone, wgt)
    return GenerationTables(wgt, tabulated_dnmax(gt, wgt), gt), N
end

# A plasma grid on a spacetime.
function mc_run(provider, plan, exec, bands, o, t0)
    rows = bands === nothing ? InclinationBands((17.0, 50.0, 90.0), 10.0) : bands
    rows isa InclinationBands ||
        throw(ArgumentError("the spectrum of a plasma grid is binned by InclinationBands; " *
                            "got $(nameof(typeof(rows)))"))
    zones = emission_zones(plan; o.zones...)
    isempty(zones) &&
        throw(ArgumentError("mc_spectrum: no emitting zones (check the matter and the cuts)"))
    g = o.gen
    jt = build_zone_emission_table(zones; n_esamp = g.n_esamp, nu_min = g.nu_min,
                                   nu_max = g.nu_max)
    wgt = build_weight_table(jt, g.Ns / (log(g.nu_max) - log(g.nu_min)))
    Nsz, dnmax = zone_counts(jt, wgt)
    lint = nothing
    g.tabulated && ((lint, dnmax) = zone_int_jnu_tables(jt, wgt))
    offset = zone_offsets(Nsz, o.seed)
    N = offset[end]
    st = plan.spacetime
    R_em = max(o.march.R_em, maximum(z -> 1.2 * proxy_distance(st, z.x)[2], zones))
    ctl = CurvedMarch(st; C = o.march.C, escape = o.march.escape, R_em = R_em)
    transport = MCTransport(ctl, st, provider, mc_opacities(plan.params, provider, o),
                            Val(o.absorption), Val(o.scattering))
    orders = (ExactBins(rows; o.grid...),)
    ledger = mc_ledger()
    layout = MCStreamLayout(max(1, N), CURVED_MAX_STEPS, o.K)
    run_layers(exec, ZoneSource(zones, wgt, dnmax, offset, lint), transport, o.seed, N,
               orders, ledger, layout, o.scattering, o.capacity)
    spec = Spectrum(orders[1])
    nb = length(rows.lo)
    mec2 = ME * CL * CL
    return GRMHDSpectrumResult(collect(rows.incl_deg), rows.half_width_deg, spec.dOmega[1:nb],
                               [band(spec, k) for k in 1:nb], band(spec, nb + 1),
                               mec2 * ledger_sum(ledger, EMITTED), mec2 * ledger_sum(ledger, ESCAPED),
                               length(zones), N, ledger_count(ledger, ESCAPED), o.scattering,
                               plan, run_record(exec, plan, t0))
end

# Cumulative primary counts of the zones: zone `z` gets ⌊N_z⌋ primaries plus one with
# probability N_z − ⌊N_z⌋, drawn in zone order from the stream `MCRng(seed, 0)`.
function zone_offsets(Nsz::AbstractVector{Float64}, seed::UInt32)
    offset = zeros(Int, length(Nsz) + 1)
    rr = MCRng(seed, 0)
    for z in eachindex(Nsz)
        fl = floor(Int, Nsz[z])
        u, rr = mc_uniform(rr)
        offset[z + 1] = offset[z] + fl + (u < (Nsz[z] - fl) ? 1 : 0)
    end
    return offset
end

# The layered loop on `exec`, as a cascade: `MCPrimary` over the primaries `1:n0`, then
# up to `layout.max_layer` layers of `MCLayer` over the children of the layer before,
# into the run totals `orders` and `ledger`, gathered here. Without scattering only the
# primaries run. Returns the number of children.
function run_layers(exec, source, transport, seed::UInt32, n0::Int, orders, ledger, layout,
                    scatter::Bool, capacity)
    slot = scatter ? ChildBuffer(Superphoton, 0, layout) : nothing
    sink = MCSink(transport.ctl, orders, slot, ledger)
    nchildren = Execution.foreach_cascade!(exec, MCPrimary(source, transport, sink, seed),
                                           MCLayer(Superphoton[], MCRng[], transport, sink),
                                           n0, scatter ? layout.max_layer : 0; capacity)
    foreach(gather!, orders)
    gather!(ledger)
    return nchildren
end
