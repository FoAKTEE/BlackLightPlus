# Block refinement indicators of adaptive image-plane rendering: pure functions of one
# block's intensity matrix that decide whether the block is replaced by four children
# at twice its pixel density. An indicator is a value; any callable `f(I) -> Bool` is
# accepted in its place. Reference: White 2022, ApJS 262, 28, adaptive ray tracing.

"""
    RefinementIndicator

Supertype of the built-in block refinement indicators: [`RelativeGradient`](@ref),
[`RelativeLaplacian`](@ref), [`AbsoluteGradient`](@ref) and the disjunction
[`AnyOf`](@ref). An indicator decides through [`needs_refinement`](@ref); wherever an
indicator is accepted, a callable `f(I::AbstractMatrix) -> Bool` is accepted too.

# References
White (2022), ApJS 262, 28: adaptive ray tracing.
"""
abstract type RefinementIndicator end

"""
    RelativeGradient(cut = 0.15; frac = 0.03)

The relative-gradient indicator: fires when more than the fraction `frac` of a block's
pixels have a relative gradient above `cut`.

At pixel `(i, j)` of an `ni × nj` block `I` the difference along `j` is the central
difference over the stencil-weighted sum,
`qx = 2 (I[i,j+1] − I[i,j−1]) / (I[i,j−1] + 2 I[i,j] + I[i,j+1])`, and two-point at the
block edges, `qx = 2 (I[i,2] − I[i,1]) / (I[i,1] + I[i,2])` at `j = 1` and
`qx = 2 (I[i,nj] − I[i,nj−1]) / (I[i,nj−1] + I[i,nj])` at `j = nj`; `qy` is the same
along `i`, and `q = hypot(qx, qy)` is dimensionless, per pixel step. A pixel with a
non-finite `q` (zero denominator, NaN or infinite intensity) is not counted. With `nch`
the counted pixels and `nex` those with `q > cut`, the indicator fires when `nch > 0`
and `nex / nch > frac`. A negative or NaN `frac` disables it. The block needs at least
two pixels along each side.

Fields: `cut`, `frac` (`Float64`).
"""
struct RelativeGradient <: RefinementIndicator
    cut::Float64
    frac::Float64
end
RelativeGradient(cut::Real = 0.15; frac::Real = 0.03) = RelativeGradient(Float64(cut), Float64(frac))

"""
    RelativeLaplacian(cut = 1.0; frac = 0.03)

The relative-Laplacian indicator: fires when more than the fraction `frac` of a block's
interior pixels have a relative Laplacian above `cut`.

At each interior pixel (`2 ≤ i ≤ ni − 1`, `2 ≤ j ≤ nj − 1`) the five-point Laplacian
`L = I[i,j−1] + I[i,j+1] + I[i−1,j] + I[i+1,j] − 4 I[i,j]` is divided by the local
weighted mean `m = (I[i,j−1] + I[i,j+1] + I[i−1,j] + I[i+1,j] + 4 I[i,j]) / 8`, giving
`q = |L| / m`, dimensionless. Counting, threshold and disabling are those of
[`RelativeGradient`](@ref); a block without interior pixels never fires.

Fields: `cut`, `frac` (`Float64`).
"""
struct RelativeLaplacian <: RefinementIndicator
    cut::Float64
    frac::Float64
end
RelativeLaplacian(cut::Real = 1.0; frac::Real = 0.03) = RelativeLaplacian(Float64(cut), Float64(frac))

"""
    AbsoluteGradient(cut; count = 5)

The value-based absolute-gradient indicator: fires when at least `count` pixels of the
block have an absolute gradient above `cut`.

At pixel `(i, j)`, `qx = (I[i,j+1] − I[i,j−1]) / 2` in the interior and the one-sided
`I[i,2] − I[i,1]`, `I[i,nj] − I[i,nj−1]` at the edges; `qy` is the same along `i`, and
`q = hypot(qx, qy)` is in intensity units per pixel step. Pixels are scanned with `i`
outer and `j` inner; a non-finite `q` is skipped, and after every finite one the count
of `q > cut` is compared with `count`, so a `count ≤ 0` fires at the first finite
pixel. Because the cut is absolute, a smooth feature falls below it under refinement
while an unresolved edge stays above it; it does not saturate on dim backgrounds as the
relative indicators do. The block needs at least two pixels along each side.

Fields: `cut::Float64`, `count::Int`.
"""
struct AbsoluteGradient <: RefinementIndicator
    cut::Float64
    count::Int
end
AbsoluteGradient(cut::Real; count::Integer = 5) = AbsoluteGradient(Float64(cut), Int(count))

"""
    AnyOf(indicators...)

The disjunction of `indicators`: fires when any of them fires, evaluated in the given
order and stopping at the first that fires. Each entry is a
[`RefinementIndicator`](@ref) or a callable `f(I) -> Bool`.

Field: `indicators::Tuple`.
"""
struct AnyOf{T<:Tuple} <: RefinementIndicator
    indicators::T
end
AnyOf(indicators...) = AnyOf(indicators)

"""
    refinement_indicator(; rel_grad_cut = 0.15, rel_grad_frac = 0.03, rel_lapl_cut = 1.0,
                         rel_lapl_frac = 0.03, abs_grad_cut = nothing, abs_grad_count = 5)
        -> RefinementIndicator

The indicator named by a keyword set: `AbsoluteGradient(abs_grad_cut; count =
abs_grad_count)` when `abs_grad_cut` is a number, the relative pair then being unused;
otherwise `AnyOf(RelativeGradient(rel_grad_cut; frac = rel_grad_frac),
RelativeLaplacian(rel_lapl_cut; frac = rel_lapl_frac))`, where a negative fraction
switches its member off. Without keywords it is the default indicator of adaptive
rendering.
"""
function refinement_indicator(; rel_grad_cut::Real = 0.15, rel_grad_frac::Real = 0.03,
                              rel_lapl_cut::Real = 1.0, rel_lapl_frac::Real = 0.03,
                              abs_grad_cut::Union{Nothing,Real} = nothing,
                              abs_grad_count::Integer = 5)
    abs_grad_cut === nothing ||
        return AbsoluteGradient(abs_grad_cut; count = abs_grad_count)
    return AnyOf(RelativeGradient(rel_grad_cut; frac = rel_grad_frac),
                 RelativeLaplacian(rel_lapl_cut; frac = rel_lapl_frac))
end

"""
    needs_refinement(indicator, I::AbstractMatrix) -> Bool

Whether the block whose pixel intensities are `I` (`I[i, j]`, pixel `i` along the first
image axis and `j` along the second, in any consistent intensity unit) is refined under
`indicator`: a [`RefinementIndicator`](@ref), or any callable, which is called as
`indicator(I)` and must return a `Bool`. The built-in indicators are pure functions of
`I`, so blocks may be tested concurrently and in any order.
"""
needs_refinement(f, I::AbstractMatrix) = f(I)::Bool

function needs_refinement(ind::RelativeGradient, I::AbstractMatrix)
    ind.frac >= 0 || return false
    ni, nj = _gradient_extent(I)
    nex = 0; nch = 0
    for i in 1:ni, j in 1:nj
        q = _relative_gradient(I, i, j, ni, nj)
        isfinite(q) || continue
        nch += 1
        q > ind.cut && (nex += 1)
    end
    return nch > 0 && nex / nch > ind.frac
end

function needs_refinement(ind::RelativeLaplacian, I::AbstractMatrix)
    ind.frac >= 0 || return false
    ni, nj = size(I)
    nex = 0; nch = 0
    for i in 2:ni-1, j in 2:nj-1
        q = _relative_laplacian(I, i, j)
        isfinite(q) || continue
        nch += 1
        q > ind.cut && (nex += 1)
    end
    return nch > 0 && nex / nch > ind.frac
end

function needs_refinement(ind::AbsoluteGradient, I::AbstractMatrix)
    ni, nj = _gradient_extent(I)
    nex = 0
    for i in 1:ni, j in 1:nj
        q = _absolute_gradient(I, i, j, ni, nj)
        isfinite(q) || continue
        q > ind.cut && (nex += 1)
        nex >= ind.count && return true
    end
    return false
end

needs_refinement(ind::AnyOf, I::AbstractMatrix) = _any_fires(ind.indicators, I)

_any_fires(::Tuple{}, I) = false
_any_fires(inds::Tuple, I) = needs_refinement(first(inds), I) || _any_fires(Base.tail(inds), I)

# The gradient stencils read a neighbour along each axis at every pixel.
function _gradient_extent(I::AbstractMatrix)
    ni, nj = size(I)
    (ni >= 2 && nj >= 2) ||
        throw(ArgumentError("a gradient indicator needs at least 2 × 2 pixels, got $ni × $nj"))
    return ni, nj
end

# Relative gradient at (i, j): averaged-denominator central differences in the
# interior, two-point differences at the block edges.
@inline function _relative_gradient(I, i, j, ni, nj)
    @inbounds begin
        qx = if j == 1
            2 * (I[i, 2] - I[i, 1]) / (I[i, 1] + I[i, 2])
        elseif j == nj
            2 * (I[i, nj] - I[i, nj-1]) / (I[i, nj-1] + I[i, nj])
        else
            2 * (I[i, j+1] - I[i, j-1]) / (I[i, j-1] + 2 * I[i, j] + I[i, j+1])
        end
        qy = if i == 1
            2 * (I[2, j] - I[1, j]) / (I[1, j] + I[2, j])
        elseif i == ni
            2 * (I[ni, j] - I[ni-1, j]) / (I[ni-1, j] + I[ni, j])
        else
            2 * (I[i+1, j] - I[i-1, j]) / (I[i-1, j] + 2 * I[i, j] + I[i+1, j])
        end
    end
    return hypot(qx, qy)
end

# Absolute gradient at (i, j): central differences in the interior, one-sided at the
# block edges, in intensity per pixel step.
@inline function _absolute_gradient(I, i, j, ni, nj)
    @inbounds begin
        qx = if j == 1
            I[i, 2] - I[i, 1]
        elseif j == nj
            I[i, nj] - I[i, nj-1]
        else
            (I[i, j+1] - I[i, j-1]) / 2
        end
        qy = if i == 1
            I[2, j] - I[1, j]
        elseif i == ni
            I[ni, j] - I[ni-1, j]
        else
            (I[i+1, j] - I[i-1, j]) / 2
        end
    end
    return hypot(qx, qy)
end

# Five-point Laplacian at the interior pixel (i, j) over the local weighted mean.
@inline function _relative_laplacian(I, i, j)
    @inbounds begin
        lap = I[i, j-1] + I[i, j+1] + I[i-1, j] + I[i+1, j] - 4 * I[i, j]
        den = (I[i, j-1] + I[i, j+1] + I[i-1, j] + I[i+1, j] + 4 * I[i, j]) / 8
    end
    return abs(lap) / den
end
