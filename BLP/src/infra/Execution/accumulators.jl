# Accumulators that per-item programs write through atomics on either executor: exact
# spectral bins and exact scalars (integer limbs, so the totals are independent of the
# executor, thread count, device count and chunking), and the child buffer of the
# layered Monte Carlo scattering loop. Under a `ThreadedExecutor` the program writes the
# host arrays. Under a `PoolExecutor` each device that runs a chunk gets a zeroed twin
# (its `DeviceAdaptor` method), registered with the host accumulator; `gather!` adds the
# twins into the host arrays after `foreach_item!` and releases them.

using ..Numerics: CL, ME, HPL, XSUM_NSLOT, XSUM_MAX_ADDENDS, xsum_atomic_add!, xsum_round,
                  xsum_round_all, xsum_zeros, MCRng, MCStreamLayout, child_rng
import Atomix

# The device twins of one host accumulator, awaiting `gather!`.
struct Twins
    lock::ReentrantLock
    list::Vector{Any}
end
Twins() = Twins(ReentrantLock(), Any[])

# Register the twin `t` of `dev`; returns `t`.
function register!(tw::Twins, dev, t)
    @lock tw.lock push!(tw.list, (dev, t))
    return t
end

# Take every registered (device, twin) pair, leaving none.
take_twins!(tw::Twins) = @lock tw.lock begin
    l = copy(tw.list)
    empty!(tw.list)
    l
end

# A zeroed device copy of the host array `a`.
zeroed(dev, a::Array{T}) where {T} = upload(dev, zeros(T, size(a)))

"""
    Bands

Angular binning rule of [`ExactBins`](@ref): [`PolarBands`](@ref) or
[`InclinationBands`](@ref).
"""
abstract type Bands end

"""
    PolarBands(n)

`n` equal bands of the polar angle about x³ folded about the equator, `θ ∈ [0, π/2]`,
`|cos θ| = |d³|/|d|` for the escape direction `d`; a direction with `|d| = 0` counts as
equatorial. Band `j` spans `[(j − 1)Δθ, jΔθ]`, `Δθ = (π/2)/n`, solid angle
`2 · 2π (cos θ_{j−1} − cos θ_j)`; the bands partition the sky (the sphere spectra,
gpumonty `report_spectrum`).
"""
struct PolarBands <: Bands
    n::Int
    dth::Float64
end
PolarBands(n::Integer) = PolarBands(Int(n), (0.5 * π) / n)

"""
    InclinationBands(incl_deg, half_width_deg)

Viewing-inclination bands of the unfolded polar angle `θ = acos(d³/|d|) ∈ [0, π]` of the
escape direction `d`: band `k` is `[θ_k − Δ, θ_k + Δ]` clipped to `[0, π]`, with
`θ_k = incl_deg[k]` and `Δ = half_width_deg` (degrees), solid angle
`2π (cos lo − cos hi)`. Bands may overlap; a photon is recorded in every band that
contains it and in the all-sky row `N + 1`; a direction with `|d| ≤ 0` is not recorded.
Fields: the definition `incl_deg`, `half_width_deg` and the clipped edges `lo`, `hi`
(radians).
"""
struct InclinationBands{N} <: Bands
    incl_deg::NTuple{N,Float64}
    half_width_deg::Float64
    lo::NTuple{N,Float64}
    hi::NTuple{N,Float64}
end
function InclinationBands(incl_deg, half_width_deg::Real)
    cen = Tuple(Float64.(incl_deg))
    hw = deg2rad(Float64(half_width_deg))
    return InclinationBands(cen, Float64(half_width_deg),
                            map(c -> clamp(deg2rad(c) - hw, 0.0, π), cen),
                            map(c -> clamp(deg2rad(c) + hw, 0.0, π), cen))
end

"Rows of the bins of a rule."
nrows(b::PolarBands) = b.n
nrows(::InclinationBands{N}) where {N} = N + 1

"Solid angle (sr) of every row of a rule."
function solid_angles(b::PolarBands)
    return [2.0 * 2.0 * π * (cos((j - 1) * b.dth) - cos(j * b.dth)) for j in 1:b.n]
end
solid_angles(b::InclinationBands) =
    [[2.0 * π * (cos(b.lo[k]) - cos(b.hi[k])) for k in eachindex(b.lo)]; 4.0 * π]

"Row of the all-sky total, 0 when the rows partition the sky."
total_row(::PolarBands) = 0
total_row(::InclinationBands{N}) where {N} = N + 1

"""
    ExactBins(bands::Bands; nu_min = 1e8, nu_max = 1e16, n_ebins = 200)

Exact spectral accumulator of escaped superphotons: per (row of `bands`, frequency
bin) the photon count and the integer limbs of `Σ w` and `Σ w E` (`E` in mₑc²).
Frequency bin `i` covers `ln ν ∈ lnu_min + [(i − 1), i) dlnu` with
`dlnu = ln(nu_max/nu_min)/n_ebins`, `ν = E mₑc²/h`; a photon outside the grid is
dropped. [`record!`](@ref) adds through atomics on host and device arrays alike, so the
bins hold the exact sums whatever the order of arrival; `merge!` adds bins of the same
grid by integer addition; [`Spectrum`](@ref) rounds them once per cell.
Inside a pool program the device twin is a zeroed accumulator gathered back by
[`gather!`](@ref); with any other `Adapt` adaptor the arrays are adapted.
"""
struct ExactBins{B<:Bands,C,A,W}
    bands::B
    nu_min::Float64
    nu_max::Float64
    lnu_min::Float64
    dlnu::Float64
    n_ebins::Int
    count::C
    sumw::A
    sumwE::A
    twins::W
end

function ExactBins(bands::Bands; nu_min::Real = 1.0e8, nu_max::Real = 1.0e16,
                   n_ebins::Integer = 200)
    lnu_min = log(nu_min)
    lnu_max = log(nu_max)
    dlnu = (lnu_max - lnu_min) / n_ebins
    nr = nrows(bands)
    return ExactBins(bands, Float64(nu_min), Float64(nu_max), lnu_min, dlnu, Int(n_ebins),
                     zeros(Int64, nr, n_ebins), xsum_zeros(nr, n_ebins),
                     xsum_zeros(nr, n_ebins), Twins())
end

"A zeroed `ExactBins` with the grid and bands of `b`."
Base.similar(b::ExactBins) =
    ExactBins(b.bands; nu_min = b.nu_min, nu_max = b.nu_max, n_ebins = b.n_ebins)

with_arrays(b::ExactBins, count, sumw, sumwE, twins) =
    ExactBins(b.bands, b.nu_min, b.nu_max, b.lnu_min, b.dlnu, b.n_ebins, count, sumw, sumwE,
              twins)

Adapt.adapt_structure(to, b::ExactBins) =
    with_arrays(b, Adapt.adapt(to, b.count), Adapt.adapt(to, b.sumw), Adapt.adapt(to, b.sumwE),
                nothing)
Adapt.adapt_structure(to::DeviceAdaptor, b::ExactBins) =
    register!(b.twins, to.dev,
              with_arrays(b, zeroed(to.dev, b.count), zeroed(to.dev, b.sumw),
                          zeroed(to.dev, b.sumwE), nothing))

"""
    record!(bins::ExactBins, E, w, dir::SVector{3}) -> nothing

Add an escaped photon of energy `E` (mₑc²), weight `w` and spatial escape direction
`dir` (not normalised): one count, `w` and `w E` into its frequency bin of every row
its direction selects. Photons with `E ≤ 0` or outside the frequency grid are dropped.
Atomic; device-safe.
"""
@inline function record!(b::ExactBins, E::Float64, w::Float64, dir::SVector{3,Float64})
    E > 0.0 || return nothing
    nu = E * ME * CL * CL / HPL
    v = floor((log(nu) - b.lnu_min) / b.dlnu)
    (v >= 0.0 && v < b.n_ebins) || return nothing
    ie = unsafe_trunc(Int, v) + 1
    record_rows!(b, b.bands, ie, E, w, dir)
    return nothing
end

@inline function record_rows!(b::ExactBins, r::PolarBands, ie::Int, E, w, d)
    ps = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    mu = ps > 0.0 ? abs(d[3]) / ps : 0.0
    thf = acos(clamp(mu, 0.0, 1.0))
    ith = clamp(unsafe_trunc(Int, floor(thf / r.dth)) + 1, 1, r.n)
    add_cell!(b, ith, ie, E, w)
    return nothing
end

@inline function record_rows!(b::ExactBins, r::InclinationBands{N}, ie::Int, E, w, d) where {N}
    ksp = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    ksp <= 0.0 && return nothing
    th = acos(clamp(d[3] / ksp, -1.0, 1.0))
    add_cell!(b, N + 1, ie, E, w)
    for k in 1:N
        (th >= r.lo[k] && th <= r.hi[k]) && add_cell!(b, k, ie, E, w)
    end
    return nothing
end

@inline function add_cell!(b::ExactBins, row::Int, ie::Int, E, w)
    Atomix.@atomic b.count[row, ie] += Int64(1)
    xsum_atomic_add!(b.sumw, w, row, ie)
    xsum_atomic_add!(b.sumwE, w * E, row, ie)
    return nothing
end

same_grid(a::ExactBins, b::ExactBins) =
    a.bands == b.bands && a.lnu_min == b.lnu_min && a.dlnu == b.dlnu && a.n_ebins == b.n_ebins

"""
    merge!(dst::ExactBins, src::ExactBins) -> dst

Add the counts and limbs of `src` (host or device arrays) into the host bins `dst` of
the same grid and bands: exact and associative.
"""
function Base.merge!(dst::ExactBins, src::ExactBins)
    same_grid(dst, src) || throw(ArgumentError("merge!: bins of different grids or bands"))
    dst.count .+= Array(src.count)
    dst.sumw .+= Array(src.sumw)
    dst.sumwE .+= Array(src.sumwE)
    return dst
end

"""
    gather!(acc) -> acc

Add the device twins that a `PoolExecutor` made of the host accumulator `acc`
([`ExactBins`](@ref), [`ExactScalars`](@ref), [`ChildBuffer`](@ref)) during
`foreach_item!` into `acc`, and release them. A no-op when there are none (a
`ThreadedExecutor` writes `acc` itself). Call it after every `foreach_item!`.
"""
function gather!(b::ExactBins)
    for (dev, t) in take_twins!(b.twins)
        merge!(b, t)
        foreach(a -> free!(dev, a), (t.count, t.sumw, t.sumwE))
    end
    return b
end

pending(acc) = @lock acc.twins.lock !isempty(acc.twins.list)

"""
    Spectrum

The host spectrum of escaped superphotons, rounded once per cell from
[`ExactBins`](@ref): `count`, `sumw` (`Σ w`) and `sumwE` (`Σ w E`, mₑc²) are
`n_bands × n_ebins` matrices; frequency bin `i` is centred at
`exp(lnu_min + (i − 1/2) dlnu)`; `dOmega[j]` is the solid angle of row `j`; `total_row`
is 0 when the rows partition the sky, else the all-sky row. `plan` and `run` are the
plan and the run record of the product that made the spectrum (`nothing` for one made
from bins directly). Accessors: [`nu_centers`](@ref), [`nuLnu`](@ref),
[`nuLnu_total`](@ref), [`nuLnu_solid`](@ref), [`total_power`](@ref),
[`total_weight`](@ref), [`mean_energy`](@ref), [`band`](@ref).
"""
struct Spectrum{P,R}
    nu_min::Float64
    nu_max::Float64
    lnu_min::Float64
    dlnu::Float64
    n_ebins::Int
    n_bands::Int
    count::Matrix{Int}
    sumw::Matrix{Float64}
    sumwE::Matrix{Float64}
    dOmega::Vector{Float64}
    total_row::Int
    plan::P
    run::R
end

"""
    Spectrum(bins::ExactBins; plan = nothing, run = nothing) -> Spectrum

The spectrum of `bins`, each cell's exact sums rounded once to nearest Float64, with the
provenance `plan` and `run`. Throws if a cell holds more than `XSUM_MAX_ADDENDS` addends
or device twins await [`gather!`](@ref).
"""
function Spectrum(b::ExactBins; plan = nothing, run = nothing)
    pending(b) && error("Spectrum: device twins pending; call gather!(bins) first")
    maximum(b.count; init = 0) <= XSUM_MAX_ADDENDS ||
        error("Spectrum: a cell exceeds $(XSUM_MAX_ADDENDS) addends")
    return Spectrum(b.nu_min, b.nu_max, b.lnu_min, b.dlnu, b.n_ebins, nrows(b.bands),
                    Matrix{Int}(b.count), xsum_round_all(b.sumw), xsum_round_all(b.sumwE),
                    solid_angles(b.bands), total_row(b.bands), plan, run)
end

"""
    band(spec::Spectrum, j) -> Spectrum

The one-row spectrum of row `j` of `spec`, with solid angle `spec.dOmega[j]` and the
provenance of `spec`.
"""
band(s::Spectrum, j::Integer) =
    Spectrum(s.nu_min, s.nu_max, s.lnu_min, s.dlnu, s.n_ebins, 1, s.count[j:j, :],
             s.sumw[j:j, :], s.sumwE[j:j, :], [s.dOmega[j]], 0, s.plan, s.run)

# The all-sky part of a spectrum: every row, or the total row.
sky(s::Spectrum) = s.total_row == 0 ? s : band(s, s.total_row)

"Frequency bin centres (Hz)."
nu_centers(s::Spectrum) = [exp(s.lnu_min + (i - 0.5) * s.dlnu) for i in 1:s.n_ebins]

"""
    nuLnu(spec) -> (nu, L)

Isotropic-equivalent `ν L_ν` (erg s⁻¹) of every row,
`L[j, i] = (mₑc²/dlnu)(4π/dΩ_j) Σ w E` (gpumonty `report_spectrum`), at the bin
centres `nu`.
"""
function nuLnu(s::Spectrum)
    nu = nu_centers(s)
    L = zeros(Float64, s.n_bands, s.n_ebins)
    pref = ME * CL * CL / s.dlnu
    @inbounds for j in 1:s.n_bands
        f = pref * 4.0 * π / s.dOmega[j]
        for i in 1:s.n_ebins
            L[j, i] = f * s.sumwE[j, i]
        end
    end
    return nu, L
end

"""
    nuLnu_total(spec) -> (nu, L, count)

All-sky `ν L_ν = (mₑc²/dlnu) Σ w E` (erg s⁻¹) summed over the rows (or the all-sky row),
`Σ_θ ν L_ν(θ) dΩ/4π`, with the photon count of each frequency bin.
"""
function nuLnu_total(spec::Spectrum)
    s = sky(spec)
    nu = nu_centers(s)
    L = zeros(Float64, s.n_ebins)
    cnt = zeros(Int, s.n_ebins)
    pref = ME * CL * CL / s.dlnu
    @inbounds for i in 1:s.n_ebins
        a = 0.0
        c = 0
        for j in 1:s.n_bands
            a += s.sumwE[j, i]
            c += s.count[j, i]
        end
        L[i] = pref * a
        cnt[i] = c
    end
    return nu, L, cnt
end

"""
    nuLnu_solid(spec, dOmega) -> (nu, L, count)

`nuLnu_total` scaled by `4π/dOmega`: the isotropic-equivalent `ν L_ν` that an observer
infers from the photons of `spec` escaping into the solid angle `dOmega` (sr), the
per-inclination convention of grmonty `report_spectrum`.
"""
function nuLnu_solid(spec::Spectrum, dOmega::Real)
    nu, L, cnt = nuLnu_total(spec)
    f = 4.0 * π / dOmega
    @inbounds for i in eachindex(L)
        L[i] *= f
    end
    return nu, L, cnt
end

"Recorded power `mₑc² Σ w E` (erg s⁻¹) over the sky."
total_power(spec::Spectrum) = ME * CL * CL * sum(sky(spec).sumwE)

"Recorded weight `Σ w` over the sky."
total_weight(spec::Spectrum) = sum(sky(spec).sumw)

"Weight-averaged energy `Σ w E / Σ w` (mₑc²) over the sky, 0 for an empty spectrum."
function mean_energy(spec::Spectrum)
    s = sky(spec)
    sw = sum(s.sumw)
    return sw > 0.0 ? sum(s.sumwE) / sw : 0.0
end

"""
    ExactScalars(n)

`n` exact scalar sums (diagnostic totals) with their addend counts. [`add!`](@ref)
adds through atomics on either executor, `merge!` adds two by integer addition,
[`scalar`](@ref) rounds one. Pool twins as for [`ExactBins`](@ref).
"""
struct ExactScalars{A,N,W}
    acc::A
    n::N
    twins::W
end
ExactScalars(n::Integer) = ExactScalars(xsum_zeros(Int(n)), zeros(Int64, n), Twins())

Adapt.adapt_structure(to, s::ExactScalars) =
    ExactScalars(Adapt.adapt(to, s.acc), Adapt.adapt(to, s.n), nothing)
Adapt.adapt_structure(to::DeviceAdaptor, s::ExactScalars) =
    register!(s.twins, to.dev, ExactScalars(zeroed(to.dev, s.acc), zeroed(to.dev, s.n), nothing))

"""
    add!(s::ExactScalars, k, x::Float64) -> nothing

Add `x` to scalar `k`. Atomic; device-safe.
"""
@inline function add!(s::ExactScalars, k::Int, x::Float64)
    xsum_atomic_add!(s.acc, x, k)
    Atomix.@atomic s.n[k] += Int64(1)
    return nothing
end

"Add the sums and addend counts of `src` into the host scalars `dst`."
function Base.merge!(dst::ExactScalars, src::ExactScalars)
    dst.acc .+= Array(src.acc)
    dst.n .+= Array(src.n)
    return dst
end

function gather!(s::ExactScalars)
    for (dev, t) in take_twins!(s.twins)
        merge!(s, t)
        foreach(a -> free!(dev, a), (t.acc, t.n))
    end
    return s
end

"""
    scalar(s::ExactScalars, k) -> Float64
    scalar(s::ExactScalars, ks::AbstractVector{<:Integer}) -> Float64

Scalar `k` rounded once to nearest Float64; with a list `ks`, the exact sum of the
scalars `ks` (the stripes of one quantity, kept in several cells so that concurrent
atomics hit different addresses), rounded once. Throws when the sum holds more than `XSUM_MAX_ADDENDS` addends or device
twins await [`gather!`](@ref).
"""
function scalar(s::ExactScalars, k::Integer)
    pending(s) && error("scalar: device twins pending; call gather!(scalars) first")
    s.n[k] <= XSUM_MAX_ADDENDS || error("exact scalar $k exceeds $(XSUM_MAX_ADDENDS) addends")
    return xsum_round(view(s.acc, :, k))
end

function scalar(s::ExactScalars, ks::AbstractVector{<:Integer})
    pending(s) && error("scalar: device twins pending; call gather!(scalars) first")
    addends(s, ks) <= XSUM_MAX_ADDENDS ||
        error("exact scalars $ks exceed $(XSUM_MAX_ADDENDS) addends")
    return xsum_round(vec(sum(view(s.acc, :, ks); dims = 2)))
end

"Number of addends of scalar `k`, or of the scalars `ks` together."
addends(s::ExactScalars, k::Integer) = Int(s.n[k])
addends(s::ExactScalars, ks::AbstractVector{<:Integer}) = Int(sum(view(s.n, ks)))

"""
    ChildBuffer(T, capacity, layout::MCStreamLayout)

Scattered children of one layer of the layered scattering loop, as a structure of
arrays: `photons::Vector{T}` and their streams `rngs::Vector{MCRng}`, `capacity` slots,
and `count[1]`, the atomic slot counter of spawn attempts. [`spawn!`](@ref) takes the
next slot from the counter and writes the child and its genealogy stream
`child_rng(rng, event, layout)` if the slot exists; a child beyond the capacity is
counted and dropped, and [`overflow`](@ref) reports how many. Children are therefore
stored in arrival order, which is not deterministic; their set is.

Policy (the driver's): a layer whose buffer overflowed is rerun from the same inputs
with a buffer of capacity at least `count[1]` (the rerun spawns the identical children,
their streams being genealogy-keyed), so no child is ever lost. Read the children with
[`children`](@ref) (copies, safe to hand to the next layer's program) and reset the
buffer with `empty!`. Inside a pool program each device writes a twin buffer of the same
capacity; [`gather!`](@ref) appends the stored children of every twin and adds the
attempt counts.
"""
struct ChildBuffer{P,R,C,W}
    photons::P
    rngs::R
    count::C
    layout::MCStreamLayout
    twins::W
end
ChildBuffer(::Type{T}, capacity::Integer, layout::MCStreamLayout) where {T} =
    ChildBuffer(Vector{T}(undef, capacity), Vector{MCRng}(undef, capacity), zeros(Int64, 1),
                layout, Twins())

Adapt.adapt_structure(to, b::ChildBuffer) =
    ChildBuffer(Adapt.adapt(to, b.photons), Adapt.adapt(to, b.rngs), Adapt.adapt(to, b.count),
                b.layout, nothing)
function Adapt.adapt_structure(to::DeviceAdaptor, b::ChildBuffer)
    bk = backend(to.dev)
    cap = length(b.photons)
    t = ChildBuffer(KernelAbstractions.allocate(bk, eltype(b.photons), cap),
                    KernelAbstractions.allocate(bk, MCRng, cap), zeroed(to.dev, b.count),
                    b.layout, nothing)
    return register!(b.twins, to.dev, t)
end

"""
    spawn!(buf::ChildBuffer, child, rng::MCRng, event) -> nothing

Store `child` with the stream `child_rng(rng, event, buf.layout)` (`rng` the parent's
stream, `event` the 0-based spawn ordinal along its march) in the next free slot, or
count it as overflow. Atomic; device-safe.
"""
@inline function spawn!(b::ChildBuffer, child, rng::MCRng, event::Integer)
    slot = Atomix.@atomic b.count[1] += Int64(1)
    if slot <= length(b.photons)
        @inbounds b.photons[slot] = child
        @inbounds b.rngs[slot] = child_rng(rng, event, b.layout)
    end
    return nothing
end

"Slots of the buffer."
capacity(b::ChildBuffer) = length(b.photons)

"Children stored in the buffer."
Base.length(b::ChildBuffer) = min(Int(b.count[1]), length(b.photons))

"Children counted but not stored for lack of slots."
overflow(b::ChildBuffer) = Int(b.count[1]) - length(b)

"""
    children(buf::ChildBuffer) -> (photons::Vector, rngs::Vector)

Copies of the stored children and their streams, in slot order.
"""
function children(b::ChildBuffer)
    pending(b) && error("children: device twins pending; call gather!(buffer) first")
    n = length(b)
    return b.photons[1:n], b.rngs[1:n]
end

"Forget every child and twin: the counter returns to zero."
function Base.empty!(b::ChildBuffer)
    take_twins!(b.twins)
    b.count[1] = 0
    return b
end

function gather!(b::ChildBuffer)
    n_att = Int(b.count[1])
    stored = length(b)
    for (dev, t) in take_twins!(b.twins)
        nd = Int(Array(t.count)[1])
        sd = min(nd, length(t.photons))
        if stored + sd > length(b.photons)
            resize!(b.photons, stored + sd)
            resize!(b.rngs, stored + sd)
        end
        if sd > 0
            copyto!(b.photons, stored + 1, Array(view(t.photons, 1:sd)), 1, sd)
            copyto!(b.rngs, stored + 1, Array(view(t.rngs, 1:sd)), 1, sd)
        end
        stored += sd
        n_att += nd
        foreach(a -> free!(dev, a), (t.photons, t.rngs, t.count))
    end
    b.count[1] = n_att
    return b
end
