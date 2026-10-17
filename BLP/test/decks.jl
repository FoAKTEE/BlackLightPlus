"""
    Decks

Shared synthetic inputs of the test suite: a torus-like primitive field on a
uniform grid (`uniform_deck`) and on a one-level block mesh (`amr_deck`), the reference run configuration,
`bits`, the bit view of a result used for bitwise comparisons, the camera rays of
every pixel, the fixed-control sampled trajectories of a ray set, the torus as a grid
provider and the program record the transfer marches read, the host-device lists of
the multi-device tests, `device_isbits`, the check that a value is a valid kernel
argument once adapted to a device, and the Monte Carlo pieces a driver assembles: a sink
over the exact accumulators, the primary and layer programs, and the sphere runs.
"""
module Decks

using StaticArrays
using BLP.Execution: HostDevice
using BLP.Data: UniformGrid, AthenaKSnapshot
using BLP.Spacetime: CameraFrame, pixel_centres, npixels, pixel_coords, camera_ray,
                     MarchBudget, march_control, rk4_step
using BLP.Matter: GridProvider, PlasmaPrimitives
import BLP.Matter: primitives, domain_radius

# BLP dependencies loaded by identity, so the module also loads in environments (the
# oracle's) where they are not direct dependencies.
const Adapt = Base.require(Base.PkgId(Base.UUID("79e6a3ab-5dfb-504d-930d-738a2a938a0e"), "Adapt"))
const ForwardDiff = Base.require(Base.PkgId(Base.UUID("f6369f11-7733-5829-9624-2563aa707210"),
                                            "ForwardDiff"))

"Bit pattern of every number in `x`; equal patterns mean bitwise-identical output, NaN and -0.0 included. A `ForwardDiff.Dual` contributes its value and then its partials."
bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
bits(x::ForwardDiff.Dual) = vcat(bits(ForwardDiff.value(x)), bits(collect(ForwardDiff.partials(x))))
bits(x::AbstractArray{<:ForwardDiff.Dual}) = reduce(vcat, (bits(v) for v in x))
bits(x::AbstractArray{<:SVector}) = reduce(vcat, (bits(collect(v)) for v in x))
bits(x::NamedTuple) = reduce(vcat, (bits(v) for v in values(x)))
bits(x::Tuple) = reduce(vcat, (bits(v) for v in x))
bits(x::Real) = [reinterpret(UInt64, Float64(x))]
bits(x::AbstractVector{<:AbstractArray}) = reduce(vcat, (bits(m) for m in x))

"Initial states of every pixel of `cam` in camera order (β fastest); errors when a ray has no positive-energy null root."
function camera_rays(st, cam)
    fr = CameraFrame(cam)
    centres = pixel_centres(cam)
    return map(1:npixels(cam)) do k
        y, ok = camera_ray(st, fr, pixel_coords(centres, cam.npix, k)...)
        ok || error("no positive-energy null root at the camera")
        y
    end
end

"""
Fixed-control trajectories of `rays` under `march_control` and `rk4_step`, ordered source
to camera: `(states, dls)` with `dls[i][k]` the affine step from state k to k+1 (last 0).
"""
function sampled_trace(st, rays::Vector{SVector{8,T}}; C, escape, nmax, R_em = T(Inf)) where {T}
    b = MarchBudget{T}(T(C), T(escape), T(R_em), nmax)
    states = Vector{Vector{SVector{8,T}}}(undef, length(rays))
    dls = Vector{Vector{T}}(undef, length(rays))
    for i in eachindex(rays)
        ys = [rays[i]]; hs = T[]
        y = rays[i]; r_prev = T(Inf)
        for _ in 1:nmax
            m = march_control(st, y, r_prev, b)
            m.stop && break
            r_prev = m.R
            y = rk4_step(st, y, m.h)
            push!(ys, y); push!(hs, m.h)
        end
        states[i] = reverse(ys)
        dls[i] = vcat(reverse(hs), zero(T))
    end
    return states, dls
end

"Torus-like primitives (rho, uu1..3, eint, bb1..3) at (x, y, z); zero outside 2.2 < r < 11."
function torus_primitives(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < 2.2 || r > 11.0) && return ntuple(_ -> 0.0f0, 8)
    rho = exp(-((r - 5.5) / 2.5)^2)
    return Float32.((rho, -0.2y / r, 0.2x / r, 0.01, 0.08rho,
                     0.3sqrt(rho), 0.1sqrt(rho) * x / r, 0.2sqrt(rho)))
end

"""
    uniform_deck(; n = 24, bh = 12.0) -> UniformGrid

The torus primitives on the `n³` nodes of `[-bh, bh]³`.
"""
function uniform_deck(; n = 24, bh = 12.0)
    data = zeros(Float32, 8, n, n, n)
    dx = 2bh / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        data[:, i, j, k] .= torus_primitives(-bh + (i - 1) * dx, -bh + (j - 1) * dx,
                                             -bh + (k - 1) * dx)
    end
    return UniformGrid(data, n, -bh, (n - 1) / (2bh), bh)
end

"""
    amr_deck(; nc = 6, half = 12.0, scale = 1.0f0) -> AthenaKSnapshot

The torus primitives on 2×2×2 level-0 blocks of `nc³` cells tiling `[-half, half]³`
(aligned to the origin, so the locate is a `LeafTable`); `scale` multiplies the density,
a second frame on the same mesh.
"""
function amr_deck(; nc = 6, half = 12.0, scale = 1.0f0)
    w = half
    dx = w / nc
    nmb = 8
    data = zeros(Float32, 8, nc, nc, nc, nmb)
    lo = zeros(3, nmb); hi = zeros(3, nmb); xl = zeros(3, nmb); dxs = fill(dx, 3, nmb)
    ll = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        lo[:, m] .= l; hi[:, m] .= l .+ w; xl[:, m] .= l .+ dx / 2
        ll[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            p = torus_primitives(xl[1, m] + (i - 1) * dx, xl[2, m] + (j - 1) * dx,
                                 xl[3, m] + (k - 1) * dx)
            for q in 1:8
                data[q, i, j, k, m] = q == 1 ? scale * p[q] : p[q]
            end
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), ll)
end

"The torus of [`torus_primitives`](@ref) as a grid provider, sampled at the Float64 position."
struct TorusDeck <: GridProvider end
function primitives(::TorusDeck, x::SVector{4})
    t = torus_primitives(x[2], x[3], x[4])
    return PlasmaPrimitives{Float64}(Float64(t[1]), SVector{3,Float64}(t[2], t[3], t[4]), Float64(t[5]),
                                     SVector{3,Float64}(t[6], t[7], t[8]), t[1] > 0)
end
domain_radius(::TorusDeck) = 11.0

"A transfer program: the fields `BLP.Transfer.march` reads; `isbits` after `Adapt.adapt`."
struct MarchProgram{S,C,P,T,F}
    st::S
    control::C
    provider::P
    params::T
    frame::F
end
Adapt.@adapt_structure MarchProgram

"Device counts of the multi-device tests."
const NDEV = (1, 2, 3, 5)

"The logical devices `HostDevice(1)`, …, `HostDevice(nd)`."
hosts(nd) = [HostDevice(k) for k in 1:nd]

"""
    DeviceLike()

`Adapt` adaptor standing in for a GPU: every `Array` becomes a [`DeviceArrayLike`](@ref),
an `isbits` pointer-and-dims view of the same memory, as device arrays are inside a
kernel.
"""
struct DeviceLike end

"An `isbits` read-only array view of host memory: element pointer and dimensions."
struct DeviceArrayLike{T,N} <: AbstractArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Int}
end
Base.size(a::DeviceArrayLike) = a.dims
Base.IndexStyle(::Type{<:DeviceArrayLike}) = IndexLinear()
Base.getindex(a::DeviceArrayLike, i::Int) = unsafe_load(a.ptr, i)

Adapt.adapt_storage(::DeviceLike, a::Array{T,N}) where {T,N} =
    DeviceArrayLike{T,N}(pointer(a), size(a))

"Whether `x` is `isbits` after `Adapt.adapt` to a device, i.e. a valid kernel argument."
device_isbits(x) = isbits(Adapt.adapt(DeviceLike(), x))

# ---------------------------------------------------------------- Monte Carlo

using LinearAlgebra: det
using BLP.Numerics: primary_rng, MCRng, MCStreamLayout, mc_uniform, Units, GNEWT, CL
using BLP.Spacetime: Minkowski, KerrSchildCartesian, ks_radius
using BLP.Matter: EmissionZone, ElectronModel, PlasmaCuts, legacy_bound, sample
using BLP.Transfer: TransferParams
using BLP.MonteCarlo: Sink, Superphoton, GenerationTables, Opacities, Bias, FlatMarch,
                      CurvedMarch, escape_direction, generate, track!, table_zone,
                      zone_nsuperphotons, SPHERE_MAX_STEPS, CURVED_MAX_STEPS
import BLP.MonteCarlo: record!, spawn!, tally!
using BLP.Execution: Execution, ExactBins, ExactScalars, ChildBuffer, PolarBands,
                     InclinationBands, foreach_item!, gather!, children, overflow
using BLP.Data: build_weight_table, build_zone_emission_table, zone_counts, locate

"""
    BinsSink(ctl, orders, children, ledger)

A Monte Carlo sink over the `BLP.Execution` accumulators: an escaped photon of
scattering order `n` is recorded into `orders[min(n, K) + 1]` (`orders` a tuple of
`K + 1` `ExactBins`) with direction `escape_direction(ctl, sph)`; children go to the
`ChildBuffer` `children` (`nothing` without scattering); the `ExactScalars` `ledger`
(`nothing` for none) receives 1: spawned weight, 2: weight spawned by primaries,
3: scattering loss, 4: weight of primaries (from `tally!`), 5: escaped `Σ w E`,
6: emitted `Σ w E` (added by the programs).
"""
struct BinsSink{C,O,K,L} <: Sink
    ctl::C
    orders::O
    children::K
    ledger::L
end
Adapt.@adapt_structure BinsSink

"A ledger of the six `BinsSink` scalars."
mc_ledger() = ExactScalars(6)

@inline function record!(s::BinsSink, sph::Superphoton)
    k = min(Int(sph.nscatt), length(s.orders) - 1) + 1
    Execution.record!(s.orders[k], sph.E, sph.w, escape_direction(s.ctl, sph))
    s.ledger === nothing || Execution.add!(s.ledger, 5, sph.w * sph.E)
    return nothing
end
@inline spawn!(s::BinsSink, child, rng, event) = Execution.spawn!(s.children, child, rng, event)
@inline function tally!(s::BinsSink, sph, loss, spawned)
    s.ledger === nothing && return nothing
    Execution.add!(s.ledger, 1, Float64(spawned))
    if sph.nscatt == 0
        Execution.add!(s.ledger, 2, Float64(spawned))
        Execution.add!(s.ledger, 4, sph.w)
    end
    Execution.add!(s.ledger, 3, Float64(loss))
    return nothing
end

# Track one photon on the program's march into its sink.
@inline track_into(p, sph, rng) =
    track!(p.ctl, p.st, p.provider, p.opac, sph, rng, p.sink; absorb = p.absorb,
           scatter = p.scatter)

# Add the emitted power of a primary to the sink's ledger.
@inline emitted!(sink, sph) =
    (sink.ledger === nothing || Execution.add!(sink.ledger, 6, sph.w * sph.E); nothing)

"""
    PrimaryProgram(source, tables, ctl, st, provider, opac, sink, seed, absorb, scatter)

Per-item Monte Carlo program without outputs: item `g` draws the stream
`primary_rng(seed, g)`, generates a superphoton from `source` (a `UniformSphere` or an
`EmissionZone`) with `tables` and tracks it with `track!` into `sink`; `absorb` and
`scatter` are the `Val` switches. `isbits` after `Adapt.adapt`.
"""
struct PrimaryProgram{G,T,C,S,P,O,K,A,B}
    source::G
    tables::T
    ctl::C
    st::S
    provider::P
    opac::O
    sink::K
    seed::UInt32
    absorb::A
    scatter::B
end
Adapt.@adapt_structure PrimaryProgram

@inline function (p::PrimaryProgram)(item, i)
    sph, rng = generate(p.source, p.tables, primary_rng(p.seed, item))
    emitted!(p.sink, sph)
    track_into(p, sph, rng)
    return nothing
end

"""
    ZoneProgram(zones, weights, dnmax, offset, ctl, st, provider, opac, sink, seed, absorb, scatter)

Per-item program over the primaries of a zone list: primary `g` belongs to the zone `z`
with `offset[z] < g ≤ offset[z + 1]` and is generated from `zones[z]` with the shared
weight table and the zone's envelope `dnmax[z]` on the stream `primary_rng(seed, g)`.
"""
struct ZoneProgram{Z,W,D,F,C,S,P,O,K,A,B}
    zones::Z
    weights::W
    dnmax::D
    offset::F
    ctl::C
    st::S
    provider::P
    opac::O
    sink::K
    seed::UInt32
    absorb::A
    scatter::B
end
Adapt.@adapt_structure ZoneProgram

@inline function (p::ZoneProgram)(item, i)
    z = searchsortedfirst(p.offset, item) - 1
    tables = GenerationTables(p.weights, @inbounds(p.dnmax[z]), nothing)
    sph, rng = generate(@inbounds(p.zones[z]), tables, primary_rng(p.seed, item))
    emitted!(p.sink, sph)
    track_into(p, sph, rng)
    return nothing
end

"""
    LayerProgram(photons, rngs, ctl, st, provider, opac, sink, absorb, scatter)

Per-item program of a scattering layer: item `j` tracks the child `photons[j]` on its
stream `rngs[j]` into `sink`.
"""
struct LayerProgram{V,R,C,S,P,O,K,A,B}
    photons::V
    rngs::R
    ctl::C
    st::S
    provider::P
    opac::O
    sink::K
    absorb::A
    scatter::B
end
Adapt.@adapt_structure LayerProgram

@inline function (p::LayerProgram)(item, i)
    @inbounds track_into(p, p.photons[item], p.rngs[item])
    return nothing
end

"""
    run_layers(exec, ctl, primary, N, layer, mkorders, layout; capacity, scatter = true)
        -> (orders, ledger, layers)

The layered Monte Carlo loop on the executor `exec` under the control `ctl`: layer 0
runs `primary(sink)` over the items `1:N`, layer `n` runs `layer(photons, rngs, sink)`
over the children of layer
`n − 1`, each into fresh accumulators (`mkorders()` and a ledger) that are added to the
run totals once the layer's `ChildBuffer` did not overflow; an overflowing layer is
rerun with a buffer of the exact size. Without scattering only layer 0 runs. Returns the
merged orders, the ledger and the `(photons, rngs)` children of every layer.
"""
function run_layers(exec, ctl, primary, N, layer, mkorders, layout::MCStreamLayout;
                    capacity = max(1, N ÷ 2), scatter = true)
    orders = mkorders()
    ledger = mc_ledger()
    layers = Tuple{Vector{Superphoton},Vector{MCRng}}[]
    photons = Superphoton[]; rngs = MCRng[]
    for depth in 0:(scatter ? layout.max_layer : 0)
        n = depth == 0 ? N : length(photons)
        n == 0 && break
        cap = capacity
        while true
            lo = mkorders(); ll = mc_ledger()
            buf = scatter ? ChildBuffer(Superphoton, cap, layout) : nothing
            sink = BinsSink(ctl, lo, buf, ll)
            prog = depth == 0 ? primary(sink) : layer(photons, rngs, sink)
            foreach_item!(exec, prog, collect(1:n))
            foreach(gather!, lo); gather!(ll)
            if scatter
                gather!(buf)
                if overflow(buf) > 0
                    cap = Int(buf.count[1])
                    continue
                end
            end
            foreach(k -> merge!(orders[k], lo[k]), eachindex(orders)); merge!(ledger, ll)
            if scatter
                photons, rngs = children(buf)
                push!(layers, (photons, rngs))
            end
            break
        end
    end
    return orders, ledger, layers
end

"""
    sphere_tables(sphere, Ns; nu_min, nu_max, n_esamp = 400) -> (tables, N_gen)

The generation tables of the sphere for about `Ns` superphotons over the band (the
weight table at `Ns / ln(nu_max/nu_min)`), and the number of primaries to launch.
"""
function sphere_tables(sphere, Ns; nu_min = 1.0e8, nu_max = 1.0e16, n_esamp = 400)
    zone = table_zone(sphere)
    wgt = build_weight_table([zone], Float64(Ns) / (log(nu_max) - log(nu_min));
                             n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    return GenerationTables(wgt, dnmax), max(1, round(Int, N_target))
end

"""
    sphere_thin_bins(exec, sphere, tables, N, seed; nu_min, nu_max, n_ebins = 200, n_thbins = 6)
        -> ExactBins

The absorbing (no scattering) sphere run of `N` primaries on the executor `exec`, binned
by folded polar angle and frequency.
"""
function sphere_thin_bins(exec, sphere, tables, N, seed; nu_min = 1.0e8, nu_max = 1.0e16,
                          n_ebins = 200, n_thbins = 6)
    bins = ExactBins(PolarBands(n_thbins); nu_min = nu_min, nu_max = nu_max, n_ebins = n_ebins)
    ctl = FlatMarch(sphere)
    sink = BinsSink(ctl, (bins,), nothing, nothing)
    prog = PrimaryProgram(sphere, tables, ctl, Minkowski(), sphere, Opacities(), sink,
                          UInt32(seed), Val(true), Val(false))
    foreach_item!(exec, prog, collect(1:N))
    gather!(bins)
    return bins
end

"""
    sphere_scatter_run(exec, sphere, table, tables, N, seed; max_scatt = 3, capacity,
                       nu_min = 1e8, nu_max = 1e17, n_ebins = 140, n_thbins = 6)
        -> (orders, ledger, layers)

The layered scattering run of the sphere without absorption (`run_layers`) from `N`
primaries: per-order spectra on folded polar bands, the ledger, the children per layer.
"""
function sphere_scatter_run(exec, sphere, table, tables, N, seed; max_scatt = 3,
                            capacity = max(1, N ÷ 2), nu_min = 1.0e8, nu_max = 1.0e17,
                            n_ebins = 140, n_thbins = 6)
    ctl = FlatMarch(sphere)
    st = Minkowski()
    opac = Opacities(; table = table, bias = Bias(sphere), max_scatt = max_scatt)
    mk() = ntuple(_ -> ExactBins(PolarBands(n_thbins); nu_min = nu_min, nu_max = nu_max,
                                 n_ebins = n_ebins), max_scatt + 1)
    primary(sink) = PrimaryProgram(sphere, tables, ctl, st, sphere, opac, sink, UInt32(seed),
                                   Val(false), Val(true))
    layer(ph, rg, sink) = LayerProgram(ph, rg, ctl, st, sphere, opac, sink, Val(false), Val(true))
    layout = MCStreamLayout(N, SPHERE_MAX_STEPS, max_scatt)
    return run_layers(exec, ctl, primary, N, layer, mk, layout; capacity = capacity)
end

"Spin, hole mass (g) and density unit (g cm⁻³) of the torus Monte Carlo deck."
const TORUS_A = 0.9
const TORUS_MBH = 6.2e9 * 1.989e33
const TORUS_DENS = 1.0e-21

"""
    torus_params(; nu_obs = 2.3e11) -> TransferParams

The plasma model of the torus Monte Carlo deck: units of a 6.2e9 M☉ hole with density
unit 1e-21 g cm⁻³, the R_low = 1, R_high = 10 electron model with `use_p` and γ = 13/9,
and σ ≤ 1.
"""
function torus_params(; nu_obs = 2.3e11)
    L = GNEWT * TORUS_MBH / CL^2
    return TransferParams(Units(TORUS_MBH, TORUS_DENS * L^3),
                          ElectronModel(mu = 0.5, ne_ni = 1.0, use_p = true, rat_low = 1.0,
                                        rat_high = 10.0, gamma = 13.0 / 9.0, gamma_i = 5.0 / 3.0,
                                        gamma_e = 4.0 / 3.0, thermal_frac = 1.0),
                          PlasmaCuts(sigma = legacy_bound(Float64, -1.0, 1.0)), nu_obs)
end

"""
    torus_zones(blocks, st, params; stride = 1, r_emit_max = Inf) -> Vector{EmissionZone}

One zone per cell centre of `blocks` (every `stride`-th cell per axis) that its own block
owns, within Kerr–Schild radius `r_emit_max`, whose decoded state is valid with
`n_e, Θₑ > 0`: proper volume `√−g Δx¹Δx²Δx³ stride³ L_unit³` (cm³).
"""
function torus_zones(blocks, st, params; stride = 1, r_emit_max = Inf)
    zones = EmissionZone{Float64}[]
    Lu3 = params.units.L_unit^3
    for mb in 1:blocks.nmb
        d = blocks.dx[:, mb]
        dVcode = d[1] * d[2] * d[3] * Float64(stride)^3
        for k in 1:stride:blocks.nk, j in 1:stride:blocks.nj, i in 1:stride:blocks.ni
            xc = SVector(0.0, blocks.x_left[1, mb] + (i - 1) * d[1],
                         blocks.x_left[2, mb] + (j - 1) * d[2], blocks.x_left[3, mb] + (k - 1) * d[3])
            locate(blocks, xc[2], xc[3], xc[4]) == mb || continue
            ks_radius(st, xc) <= r_emit_max || continue
            s, g = sample(blocks, st, xc, params)
            (s.valid && s.n_e > 0.0 && s.theta_e > 0.0) || continue
            detg = det(g[1])
            push!(zones, EmissionZone(s, (detg < 0.0 ? sqrt(-detg) : 1.0) * dVcode * Lu3, xc, g))
        end
    end
    return zones
end

"""
    torus_run(exec, blocks, zones, Ns, seed; scattering = false, table = nothing, bias0 = 1.0,
              max_scatt = 3, C = 32.0, incl_deg = (17.0, 50.0, 90.0), half_width_deg = 10.0,
              n_ebins = 120, capacity) -> (bins, ledger, n_gen, layers)

The Monte Carlo of the torus deck on the Kerr hole of `TORUS_A`: the weight table of
the zones for about `Ns` primaries over [1e8, 1e16] Hz, per-zone counts rounded
stochastically on the stream `MCRng(seed, 0)`, a `CurvedMarch` whose emission radius is
the larger of 100 and 1.2 times the outermost zone radius, absorption always and
scattering on request, binned by inclination band over [1e8, 1e24] Hz.
"""
function torus_run(exec, blocks, zones, Ns, seed; scattering = false, table = nothing,
                   bias0 = 1.0, max_scatt = 3, C = 32.0, incl_deg = (17.0, 50.0, 90.0),
                   half_width_deg = 10.0, n_ebins = 120, capacity = 1024)
    st = KerrSchildCartesian(TORUS_A)
    params = torus_params()
    jt = build_zone_emission_table(zones; nu_min = 1.0e8, nu_max = 1.0e16)
    wgt = build_weight_table(jt, Float64(Ns) / (log(1.0e16) - log(1.0e8)))
    Nsz, dnmax = zone_counts(jt, wgt)
    offset = zeros(Int, length(zones) + 1)
    rr = MCRng(Int(seed), 0)
    for z in eachindex(zones)
        fl = floor(Int, Nsz[z])
        u, rr = mc_uniform(rr)
        offset[z + 1] = offset[z] + fl + (u < (Nsz[z] - fl) ? 1 : 0)
    end
    N = offset[end]
    R_em = max(100.0, maximum(z -> 1.2 * ks_radius(st, z.x), zones))
    ctl = CurvedMarch(st; C = C, escape = 1.0e3, R_em = R_em)
    opac = Opacities(params; table = table, bias = Bias(bias0), max_scatt = max_scatt)
    mk() = (ExactBins(InclinationBands(incl_deg, half_width_deg); nu_min = 1.0e8,
                      nu_max = 1.0e24, n_ebins = n_ebins),)
    sc = Val(scattering)
    primary(sink) = ZoneProgram(zones, wgt, dnmax, offset, ctl, st, blocks, opac, sink,
                                UInt32(seed), Val(true), sc)
    layer(ph, rg, sink) = LayerProgram(ph, rg, ctl, st, blocks, opac, sink, Val(true), sc)
    layout = MCStreamLayout(max(1, N), CURVED_MAX_STEPS, scattering ? max_scatt : 0)
    orders, ledger, layers = run_layers(exec, ctl, primary, N, layer, mk, layout;
                                        capacity = capacity, scatter = scattering)
    return orders[1], ledger, N, layers
end

end # module
