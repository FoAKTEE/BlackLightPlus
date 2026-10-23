# Precision lanes: the number type the geodesic is marched in, the transfer lane and the
# accumulator. A plan property, independent of where the plan runs.

"""
    Lanes{Tg,Tt,A}()
    Lanes()

Uniform precision lanes: every ray's geodesic (and, polarized, its transported camera
frame) is marched in `Tg`, the transfer in `Tt` and the image accumulated in `A`.
`Lanes()` is `Lanes{Float64,Float64,Float64}()`; `Lanes{Float32,Float64,Float64}()`
marches every geodesic in `Float32`, which fails near the critical curve (see
[`MixedLanes`](@ref)). The marches run the transfer and the accumulator in `Float64`,
so `prepare` accepts `Tt = A = Float64` only.
"""
struct Lanes{Tg,Tt,A} end
Lanes() = Lanes{Float64,Float64,Float64}()

"""
    MixedLanes(rescue_px = 2)

Mixed precision lanes on Kerr: the geodesics of the pixels within `rescue_px` pixels
(Chebyshev distance) of the critical curve, where a `Float32` march loses the ray, are
marched in `Float64` and queued first; every other pixel is marched in `Float32`. The
transfer and the accumulator are `Float64`.
"""
struct MixedLanes
    rescue_px::Int
    function MixedLanes(rescue_px::Integer = 2)
        rescue_px >= 0 || throw(ArgumentError("MixedLanes: rescue_px must be non-negative"))
        return new(Int(rescue_px))
    end
end

"""
    geodesic_type(lanes) -> Type

The number type of the geodesic lane of `lanes`: `Tg` for `Lanes{Tg}`, `Float32` (the
bulk lane) for `MixedLanes`.
"""
geodesic_type(::Lanes{Tg}) where {Tg} = Tg
geodesic_type(::MixedLanes) = Float32

Base.show(io::IO, ::Lanes{Tg,Tt,A}) where {Tg,Tt,A} = print(io, "Lanes{", Tg, ",", Tt, ",", A, "}()")
