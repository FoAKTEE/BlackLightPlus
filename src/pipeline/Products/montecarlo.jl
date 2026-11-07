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
# so the generated frequency density is uniform in ln ν, the expected number of primaries
# is `Ns` (trapezoid counts), and ν L_ν(bin) = (mₑc²/dlnu) Σ w E estimates the emitted
# spectrum with no free constant.

"""
    ScatterResult

The spectrum of the scattering sphere ([`mc_spectrum`](@ref) with `scattering = true`).
`orders[n + 1]` is the escaped `Spectrum` of the photons of scattering order `n`
(`orders[end]` holds the order `max_scatt`), `total` the exact sum of the orders; each
carries `Σ (w E)²` per bin for its standard error (`Execution.nuLnu_sigma`).
Diagnostics: `n_primary` primaries and `n_children` children tracked; in weight
(photons s⁻¹) the primaries `w_primary`, the children `w_children`, of which
`w_first_order` spawned by primaries, `scatter_loss`, the weight scattered out of the
beam in the steps that may spawn (the expectation of `w_children + w_unspawned`),
`scatter_loss_final`, that of the steps of the order `max_scatt`, which spawn nothing,
and `w_unspawned`, the weight of the events that spawned nothing; in power (erg s⁻¹,
`mₑc² Σ w E`) `emitted_power` by the primaries, `escaped_power`, `absorbed_power`,
`captured_power` and `dropped_power` (photons ended by the step limit). `plan` and `run`
as in [`Image`](@ref).
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
    scatter_loss_final::Float64
    w_unspawned::Float64
    emitted_power::Float64
    escaped_power::Float64
    absorbed_power::Float64
    captured_power::Float64
    dropped_power::Float64
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
isotropic-equivalent `ν L_ν`); `total` is the all-sky spectrum; both sum the scattering
orders, and `orders[n + 1]` is the spectrum of order `n` alone, every band and the
all-sky row (`orders[end]` the order `max_scatt`; one order without scattering). Every
spectrum carries `Σ (w E)²` per bin for its standard error (`Execution.nuLnu_sigma`).
Diagnostics: `n_zones` emitting zones, `n_gen` primaries, `n_recorded` escaped photons,
`n_children` children, `scattering`, whether Compton scattering was tracked; the
weights `w_primary`, `w_children`, `w_first_order`, `scatter_loss`, `scatter_loss_final`
and `w_unspawned` and the powers `emitted_power`, `escaped_power`, `absorbed_power`,
`captured_power` and `dropped_power` of a [`ScatterResult`](@ref) (the weights zero
without scattering). `plan` and `run` as in [`Image`](@ref).
"""
struct GRMHDSpectrumResult{S,P,R}
    incl_deg::Vector{Float64}
    half_width_deg::Float64
    dOmega::Vector{Float64}
    spectra::Vector{S}
    total::S
    orders::Vector{S}
    max_scatt::Int
    n_zones::Int
    n_gen::Int
    n_recorded::Int
    n_children::Int
    w_primary::Float64
    w_children::Float64
    w_first_order::Float64
    scatter_loss::Float64
    scatter_loss_final::Float64
    w_unspawned::Float64
    emitted_power::Float64
    escaped_power::Float64
    absorbed_power::Float64
    captured_power::Float64
    dropped_power::Float64
    scattering::Bool
    plan::P
    run::R
end

Base.show(io::IO, r::GRMHDSpectrumResult) =
    print(io, "GRMHDSpectrumResult(", length(r.spectra), " bands, ", r.n_zones, " zones, ",
          r.n_gen, " primaries, ", r.n_recorded, " escaped, ", r.run.executor.kind, ")")

# The ledger totals of a run, in the field order of the results: the six weights, then
# the five powers (erg s⁻¹).
function ledger_totals(l::ExactScalars)
    mec2 = ME * CL * CL
    return (ledger_sum(l, PRIMARY_WEIGHT), ledger_sum(l, SPAWNED),
            ledger_sum(l, SPAWNED_BY_PRIMARIES), ledger_sum(l, SCATTER_LOSS),
            ledger_sum(l, SCATTER_LOSS_FINAL), ledger_sum(l, UNSPAWNED),
            mec2 * ledger_sum(l, EMITTED), mec2 * ledger_sum(l, ESCAPED),
            mec2 * ledger_sum(l, ABSORBED), mec2 * ledger_sum(l, CAPTURED),
            mec2 * ledger_sum(l, DROPPED))
end

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
    emission_zones(plan; stride = 1, seed = 0x5eed5eed, r_emit_max = Inf,
                   max_zones = typemax(Int)) -> Vector{EmissionZone{Float64}}
    emission_zones(blocks::FluidBlocks, st, params; stride, seed, r_emit_max, max_zones)

The emission zones of the plan's matter (an `AthenaKSnapshot` or `FluidBlocks`) as
`mc_spectrum` draws them, decoded in full: `Data.emission_sites` of its blocks on the
plan's spacetime with the plan's parameter set (one zone per group of `stride³` cells,
its cell drawn uniformly among the group's owned cells on a stream of `seed`, volume
the owned cells' measure √−g Δ³x L³ at that cell; every owned cell above the horizon proxy at
`stride = 1`), each zone `Matter.emission_zone` of its site.
"""
emission_zones(plan::Plan; kw...) =
    emission_zones(zone_blocks(plan.provider), plan.spacetime, plan.params; kw...)

function emission_zones(blocks::FluidBlocks, st, params; stride::Integer = 1,
                        seed::Integer = 0x5eed5eed, r_emit_max::Real = Inf,
                        max_zones::Integer = typemax(Int))
    sites = emission_sites(blocks, st, params; stride = stride, seed = UInt32(seed & 0xffffffff),
                           r_emit_max = r_emit_max, max_zones = max_zones)
    return [emission_zone(blocks, st, s, params) for s in sites]
end

zone_blocks(b::FluidBlocks) = b
zone_blocks(s::AthenaKSnapshot) = s.blocks
zone_blocks(p) =
    throw(ArgumentError("emission zones need an AthenaKSnapshot or FluidBlocks as matter; " *
                        "got $(nameof(typeof(p)))"))

# ---------------------------------------------------------------- the product

"Default cap of the state-dependent bias of `mc_spectrum` (`bias_max`)."
const MC_BIAS_MAX = 1.0e6

"""
    mc_spectrum(plan, Ns; scattering = false, backend = nothing, bands = nothing,
                seed = 0x5eed5eed, nu_min = 1e8, nu_max = 1e16, spec_nu_min = nu_min,
                spec_nu_max = scattering ? 1e24 : nu_max, n_esamp = 400, n_ebins = 200,
                table = nothing, max_scatt = 3, bias = nothing, bias_max = MC_BIAS_MAX,
                bias_order = 1, split = 1, absorption, capacity = nothing, C = plan.budget.C,
                escape = 1e3, R_em = 100.0, stride = 1, r_emit_max = Inf,
                max_zones = typemax(Int), medium = nothing, step_frac = SPHERE_STEP_FRAC,
                window = nothing, time_bins = 64, time_ebins = 20, sectors = 0)
        -> Spectrum | ScatterResult | GRMHDSpectrumResult | TimedSpectrumResult

The Monte Carlo spectrum of the plan's matter from about `Ns` superphotons, run on the
executor `resolve(backend)` (`nothing`: the session default).

The matter decides the run. The uniform sphere of `prepare(sphere)` (flat spacetime):
primaries uniform in the ball, a straight-line march (`FlatMarch`); without scattering
the result is the absorbed `Spectrum`, with scattering a [`ScatterResult`](@ref) of
per-order spectra (`absorption` defaults to `false` there: the idealized Comptonization
sphere); `medium` is another plasma for the march (a `SphericalShell` about a small
source: a light echo), whose radius extends the march, and `step_frac` the march step as
a fraction of its radius. A plasma grid (`AthenaKSnapshot`, `FluidBlocks`) on any spacetime, a moving
binary included: the grid is one snapshot, emitting from its cell centres at `t = 0`
and held fixed while the superphotons travel, and the march reads the metric at each
photon's own time; primaries spread over the [`emission_zones`](@ref) of the plan
(every owned cell above the horizon at `stride = 1`; one randomly drawn cell per group
of `stride³` with the group's volume otherwise, an unbiased quadrature for quick runs;
`r_emit_max`, `max_zones`) by their expected counts, rounded stochastically on the stream
`MCRng(seed, 0)`; a geodesic march (`CurvedMarch`: step divisor `C`, escape at radius
`escape`, emission-confinement radius the larger of `R_em` and 1.2 times the outermost
zone radius); absorption always by default; the result a [`GRMHDSpectrumResult`](@ref).

Generation: the weight table over `[nu_min, nu_max]` on `n_esamp` intervals for about
`Ns` primaries (trapezoid counts, so their expectation is `Ns`), frequencies by rejection
against `∫ j_ν dΩ` from the process's `EmissivityTable`. Primary `g` draws the stream
`primary_rng(seed, g)`. Transport: absorption (`absorption`), Compton scattering
(`scattering`) with the hot-cross `table` (`nothing`: `hotcross_table(plan)`, the
process's table, built on the run's executor at the process's first request), at most
`max_scatt` scatterings per photon, every step's scattering events a Poisson process
under the importance `bias`: `nothing` or a number `b0` (1 by default) for the
`StateBias` `b = clamp(b0 A(Θₑ) (w/w_ref) bias_order^n, 1, bias_max)` of a photon
scattered `n` times, with `w_ref` the mean primary weight, `bias_order` the factor per
scattering order (1: none; larger samples the light scattered two and three times and
the high-frequency tail it makes) and `split` the most children per capped event (1: no
splitting), or a `Bias`/`StateBias` used as given. Every choice leaves the spectrum unbiased; the
bias and `split` set its variance and cost. Lengths in `plan.params.units.L_unit` cm (one
cm for a plan without parameters). Binning: `bands` (`PolarBands(6)` for the sphere,
`InclinationBands((17, 50, 90), 10)` on a grid, which takes inclination bands only),
`n_ebins` log-frequency bins over `[spec_nu_min, spec_nu_max]`.

Scattering runs as a cascade over chunks of primaries ([`Execution.foreach_cascade!`](@ref)):
within a chunk, each layer's children are tracked by the next layer, up to `max_scatt`
layers, and stay on the worker that runs the chunk. On devices a layer starts with a
child buffer of `capacity` slots (`nothing`: at least 4096, half the chunk's primaries
and the slots the device grew to in the run's earlier chunks); a layer whose buffer
overflows is rerun into zeroed scratch accumulators with a buffer for the counted
children, so no child is lost or counted twice. On threads each task's two child
buffers start with `capacity` slots (`nothing`: 4096) and grow; a photon whose children
overflow is rerun the same way.

Emission over a window: with `window = (t_a, t_b)` (coordinate time) the plasma emits over
the window and the result is a [`TimedSpectrumResult`](@ref): the stationary result of the
same matter, with every spectrum the window mean, and the escaped photons binned by their
arrival time `t − n̂·x` at a distant observer into `time_bins` equal bins over
`[t_a − R, t_b + R]` (`R` the march's emission radius; or `(t_lo, t_hi, n)`) with an
underflow and an overflow bin, on `time_ebins` frequency bins over
`[spec_nu_min, spec_nu_max]`, in the run's bands split into `sectors` azimuth sectors
(`Execution.SectorBands`; 0: none). The matter may be a frozen snapshot or sphere (on a
stationary spacetime), a `Data.Corotating` plasma (on a spacetime with that helical
symmetry), a `Data.FrameSequence` or `SeriesProvider` of frames (on a stationary
spacetime; the frame times in the coordinate time of the spacetime, the window inside
them) or a `Matter.VariableSphere`; the time-dependent ones need the window. The march
reads plasma and metric at each photon's time. Runs on CPU threads.

Determinism: bitwise reduction. The bins, the ledger and the children depend only on the
plan, `Ns` and the keywords, whatever the executor, the device count, the chunking or
the buffer capacity. A frozen snapshot over a window is the stationary run bit for bit
but for the photons' emission times.
"""
function mc_spectrum(plan::Plan, Ns::Real; scattering::Bool = false, backend = nothing,
                     bands = nothing, seed::Integer = 0x5eed5eed, nu_min::Real = MC_NU_MIN,
                     nu_max::Real = MC_NU_MAX, spec_nu_min::Real = nu_min,
                     spec_nu_max::Real = scattering ? 1.0e24 : nu_max,
                     n_esamp::Integer = MC_N_ESAMP, n_ebins::Integer = 200,
                     table = nothing, max_scatt::Integer = 3, bias = nothing,
                     bias_max::Real = MC_BIAS_MAX, bias_order::Real = 1.0, split::Integer = 1,
                     absorption::Bool = !(scattering && plan.provider isa Union{UniformSphere,VariableSphere}),
                     capacity = nothing, C::Real = plan.budget.C, escape::Real = 1.0e3,
                     R_em::Real = 100.0, stride::Integer = 1, r_emit_max::Real = Inf,
                     max_zones::Integer = typemax(Int), window = nothing, time_bins = 64,
                     time_ebins::Integer = 20, sectors::Integer = 0, medium = nothing,
                     step_frac::Real = SPHERE_STEP_FRAC)
    t0 = time_ns()
    exec = resolve(backend)
    Ns > 0 || throw(ArgumentError("mc_spectrum: Ns must be positive"))
    K = scattering ? Int(max_scatt) : 0
    K >= 0 || throw(ArgumentError("mc_spectrum: max_scatt must be non-negative"))
    capacity === nothing || capacity >= 1 ||
        throw(ArgumentError("mc_spectrum: capacity must be positive"))
    bias_max >= 1 || throw(ArgumentError("mc_spectrum: bias_max must be at least 1"))
    split >= 1 || throw(ArgumentError("mc_spectrum: split must be positive"))
    bias_order >= 1 || throw(ArgumentError("mc_spectrum: bias_order must be at least 1"))
    o = (seed = UInt32(seed & 0xffffffff), scattering = scattering, absorption = absorption,
         K = K,
         gen = (nu_min = Float64(nu_min), nu_max = Float64(nu_max), n_esamp = Int(n_esamp),
                Ns = Float64(Ns)),
         grid = (nu_min = Float64(spec_nu_min), nu_max = Float64(spec_nu_max),
                 n_ebins = Int(n_ebins)),
         table = scattering ? hotcross(table, plan, exec) : table,
         bias = (b = bias, b_max = Float64(bias_max), split = Int(split),
                 order = Float64(bias_order)), capacity = capacity,
         march = (C = Float64(C), escape = Float64(escape), R_em = Float64(R_em)),
         zones = (stride = Int(stride), r_emit_max = Float64(r_emit_max),
                  max_zones = Int(max_zones)),
         time = (window = window, bins = time_bins, ebins = Int(time_ebins), sectors = Int(sectors)),
         medium = medium, step_frac = Float64(step_frac))
    window === nothing && return mc_run(plan.provider, plan, exec, bands, o, t0)
    exec isa ThreadedExecutor ||
        throw(ArgumentError("mc_spectrum: a run over a window runs on CPU threads (backend = :cpu)"))
    sectors >= 0 && time_ebins >= 1 ||
        throw(ArgumentError("mc_spectrum: sectors must be non-negative and time_ebins positive"))
    return mc_run_timed(plan.provider, plan, exec, bands, o, t0)
end

# The hot-cross table of a run: the one given, or the plan's, built on the run's executor
# at the process's first request.
hotcross(::Nothing, plan, exec) = hotcross_table(plan, () -> hotcross_build(exec))
hotcross(table, plan, exec) = table

# The importance bias of a run whose mean primary weight is `w_ref`.
mc_bias(b::Nothing, kw, w_ref) = StateBias(1.0, w_ref, kw.b_max, kw.split, kw.order)
mc_bias(b::Real, kw, w_ref) = StateBias(Float64(b), w_ref, kw.b_max, kw.split, kw.order)
mc_bias(b::Union{Bias,StateBias}, kw, w_ref) = b

# The opacities of a run: lengths in the plan's units, or in cm without a parameter set.
mc_opacities(::Nothing, o, w_ref) =
    Opacities(; table = o.table, bias = mc_bias(o.bias.b, o.bias, w_ref), max_scatt = o.K)
mc_opacities(params, o, w_ref) =
    Opacities(params; table = o.table, bias = mc_bias(o.bias.b, o.bias, w_ref), max_scatt = o.K)

mc_run(::Nothing, plan, exec, bands, o, t0) =
    throw(ArgumentError("mc_spectrum needs matter: prepare the plan with `matter = …`"))

# Time-dependent matter needs the emission window.
mc_run(m::Union{VariableSphere,FrameSequence,SeriesProvider,Corotating}, plan, exec, bands, o, t0) =
    throw(ArgumentError("mc_spectrum: $(nameof(typeof(m))) changes in time; give the emission " *
                        "window, window = (t_a, t_b)"))

# The uniform sphere, the medium of its march (`medium`, the sphere by default) and the
# radius of the march, the larger of the two.
function mc_run(sphere::UniformSphere, plan, exec, bands, o, t0)
    plan.spacetime isa Minkowski ||
        throw(ArgumentError("the Monte Carlo sphere runs on Minkowski(); the plan's " *
                            "spacetime is $(nameof(typeof(plan.spacetime)))"))
    tables, N = sphere_tables(sphere, o.gen)
    medium = o.medium === nothing ? sphere : o.medium
    ctl = FlatMarch(max(Float64(sphere.R), Float64(domain_radius(medium))), o.step_frac,
                    SPHERE_MAX_STEPS)
    transport = MCTransport(ctl, plan.spacetime, medium,
                            mc_opacities(plan.params, o, mean_weight(tables.weights)),
                            Val(o.absorption), Val(o.scattering))
    rows = bands === nothing ? PolarBands(6) : bands
    orders = ntuple(_ -> ExactBins(rows; o.grid...), o.K + 1)
    ledger = mc_ledger()
    layout = widest_layout(N, o.K)
    nchildren = run_layers(exec, SphereSource(sphere, tables), transport, o.seed, N, orders,
                           ledger, layout, o.scattering, o.capacity)
    return sphere_result(orders, o, N, nchildren, ledger, plan, run_record(exec, plan, t0))
end

# The sphere's result of the run's bins: the absorbed spectrum, or the orders with scattering.
function sphere_result(orders, o, N, nchildren, ledger, plan, rec)
    o.scattering || return Spectrum(orders[1]; plan = plan, run = rec)
    total = similar(orders[1])
    foreach(b -> merge!(total, b), orders)
    return ScatterResult([Spectrum(b) for b in orders], Spectrum(total), o.K, N, nchildren,
                         ledger_totals(ledger)..., plan, rec)
end

# The one-zone generation tables of the sphere for about `Ns` primaries over the band,
# and the number of primaries.
function sphere_tables(sphere::UniformSphere, g)
    zone = table_zone(sphere)
    wgt = build_weight_table([zone], g.Ns / (log(g.nu_max) - log(g.nu_min));
                             n_esamp = g.n_esamp, nu_min = g.nu_min, nu_max = g.nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    return GenerationTables(wgt, dnmax), max(1, round(Int, N_target))
end

# A plasma grid on a spacetime.
function mc_run(provider, plan, exec, bands, o, t0)
    rows = bands === nothing ? InclinationBands((17.0, 50.0, 90.0), 10.0) : bands
    rows isa InclinationBands ||
        throw(ArgumentError("the spectrum of a plasma grid is binned by InclinationBands; " *
                            "got $(nameof(typeof(rows)))"))
    st = plan.spacetime
    sites, r_em, wgt, Nsz, dnmax = grid_zones(zone_blocks(provider), st, plan.params, exec, o)
    offset = zone_offsets(Nsz, o.seed)
    N = offset[end]
    R_em = max(o.march.R_em, r_em)
    ctl = CurvedMarch(st; C = o.march.C, escape = o.march.escape, R_em = R_em)
    transport = MCTransport(ctl, st, provider, mc_opacities(plan.params, o, mean_weight(wgt)),
                            Val(o.absorption), Val(o.scattering))
    orders = ntuple(_ -> ExactBins(rows; o.grid...), o.K + 1)
    ledger = mc_ledger()
    layout = widest_layout(max(1, N), o.K)
    source = ZoneSource(sites, provider, st, plan.params, wgt, EMISSIVITY_TABLE, dnmax, offset)
    nchildren = run_layers(exec, source, transport, o.seed, N, orders, ledger, layout,
                           o.scattering, o.capacity)
    return grid_result(orders, rows, o, length(sites), N, ledger, nchildren, plan,
                       run_record(exec, plan, t0))
end

# The grid's result of the run's bins.
function grid_result(orders, rows, o, n_zones, N, ledger, nchildren, plan, rec)
    total = similar(orders[1])
    foreach(b -> merge!(total, b), orders)
    spec = Spectrum(total)
    nb = length(rows.lo)
    return GRMHDSpectrumResult(collect(rows.incl_deg), rows.half_width_deg, spec.dOmega[1:nb],
                               [band(spec, k) for k in 1:nb], band(spec, nb + 1),
                               [Spectrum(b) for b in orders], o.K, n_zones, N,
                               ledger_count(ledger, ESCAPED), nchildren,
                               ledger_totals(ledger)..., o.scattering, plan, rec)
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
# into the run totals `orders` and `ledger` and, with `timed` (a `TimedRecord`), the
# time-resolved bins and the clock of `TimedPrimary` and `TimedLayer`, gathered here.
# Without scattering only the primaries run. Returns the number of children.
function run_layers(exec, source, transport, seed::UInt32, n0::Int, orders, ledger, layout,
                    scatter::Bool, capacity; timed = nothing)
    slot = scatter ? ChildBuffer(Superphoton, 0, layout) : nothing
    sink = MCSink(transport.ctl, orders, slot, ledger)
    gen = Execution.Generation(Superphoton[], MCRng[])
    primary = timed === nothing ? MCPrimary(source, transport, sink, seed) :
              TimedPrimary(source, transport, sink, seed, timed)
    layer = timed === nothing ? MCLayer(gen, transport, sink) : TimedLayer(gen, transport, sink, timed)
    nchildren = Execution.foreach_cascade!(exec, primary, layer, n0,
                                           scatter ? layout.max_layer : 0; capacity)
    foreach(gather!, orders)
    timed === nothing || (foreach(gather!, timed.timed); gather!(timed.clock))
    gather!(ledger)
    return nchildren
end
