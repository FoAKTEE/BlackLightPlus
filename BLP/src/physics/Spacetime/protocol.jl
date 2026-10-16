# The spacetime protocol. Conventions: signature (-,+,+,+), G = c = M = 1, Cartesian
# coordinates x^μ = (t, x, y, z). A spacetime defines `metric(st, x)`; every other
# method has a default here that a closed form may override.

"""
    Spacetime
    Spacetime{T}

The module: spacetimes and null geodesics — the `Spacetime` protocol (metric,
inverse, index operations, gradients), the geodesic and parallel-transport
right-hand sides, the fixed-step integrator, the termination rule, the camera,
Kerr orbit primitives and the analytic Kerr lensing bands.

The abstract type: a metric on Cartesian coordinates `x^μ = (t, x, y, z)` with
signature (−,+,+,+) in geometrized units. `T` is the number type of the
spacetime's own scalars (the lane: `Float64`, `Float32`, or a `ForwardDiff.Dual`
for parameter derivatives). A concrete spacetime implements
[`metric`](@ref); the index operations, the gradients, the geodesic right-hand
sides and the integrator then follow.
"""
abstract type Spacetime{T} end

"""
    StationarySpacetime{T} <: Spacetime{T}

A spacetime with `∂_t g_μν ≡ 0`: `p_t` is conserved by the geodesic flow and the
metric gradients are taken over the three spatial coordinates only.
"""
abstract type StationarySpacetime{T} <: Spacetime{T} end

"""
    ETA

The Minkowski metric `diag(-1, 1, 1, 1)` as an `SMatrix{4,4,Float64}`.
"""
const ETA = @SMatrix Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

# η in the value precision of a metric component type: Float32 stays Float32, a Dual
# takes η in its value type (η carries no derivative), anything else the Float64 ETA.
@inline _eta(::Type{T}) where {T<:AbstractFloat} = SMatrix{4,4,T}(ETA)
@inline _eta(::Type{ForwardDiff.Dual{G,V,N}}) where {G,V,N} = _eta(V)
@inline _eta(::Type) = ETA

# η^{μν} v_ν (= η_{μν} v^ν): flip the time component.
@inline _eta_apply(v::SVector{4}) = SVector(-v[1], v[2], v[3], v[4])

"""
    metric(st::Spacetime, x::SVector{4}) -> SMatrix{4,4}

The covariant metric `g_μν` at the contravariant position `x`. Required of every
spacetime; generic in the number type of `x` (a `Dual` position yields the
metric's derivatives in the partials).
"""
function metric end

"""
    inverse_metric(st::Spacetime, x::SVector{4}) -> SMatrix{4,4}

The contravariant metric `g^μν` at `x`; the default inverts `metric(st, x)`.
Closed forms override it.
"""
@inline inverse_metric(st::Spacetime, x::SVector{4}) = inv(metric(st, x))

"""
    metric_and_inverse(st::Spacetime, x::SVector{4}) -> (gcov, gcon)

The covariant and contravariant metric at `x` as a pair, the form later stages
pass on as `g`. Closed forms evaluate their shared terms once.
"""
@inline metric_and_inverse(st::Spacetime, x::SVector{4}) = (metric(st, x), inverse_metric(st, x))

"""
    lower_index(st::Spacetime, x, v::SVector{4}) -> SVector{4}

The contraction `g_μν(x) v^ν`. Closed forms skip the 4×4 matrix; with a `Dual`
position the partials are the derivatives of the contraction.
"""
@inline lower_index(st::Spacetime, x, v::SVector{4}) = metric(st, x) * v

"""
    raise_index(st::Spacetime, x, p::SVector{4}) -> SVector{4}

The contraction `g^μν(x) p_ν`. Closed forms skip the 4×4 matrix.
"""
@inline raise_index(st::Spacetime, x, p::SVector{4}) = inverse_metric(st, x) * p

"""
    raise_and_norm(st::Spacetime, x, p::SVector{4}) -> (g^μν p_ν, g^μν p_μ p_ν)

The raised covector and its squared norm (twice the geodesic Hamiltonian) from
one metric evaluation.
"""
@inline raise_and_norm(st::Spacetime, x, p::SVector{4}) = (gp = raise_index(st, x, p); (gp, dot(p, gp)))

"""
    horizon_radius(st) -> horizon proxy

The surface the march control measures capture against:
`KerrSchildCartesian` the outer horizon radius `r₊ = 1 + √(1 - a²)`;
`SuperposedKerrSchild` the per-hole radii (`horizon_radius(st, A)` for hole `A`);
`ADMGrid` the lapse floor `alpha_min`.
"""
function horizon_radius end

"""
    lane(st, ::Type{T}) -> spacetime with scalars of type `T`

The spacetime on the number-type lane `T` (e.g. the `Float32` geodesic lane):
every scalar converted from the values held in `st`, never recomputed in `T`.
Grid data is untouched.
"""
function lane end
