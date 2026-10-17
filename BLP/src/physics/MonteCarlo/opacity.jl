# Invariant opacity rates along a superphoton, ν α_ν and ν n_e σ_h (Hz cm⁻¹), from the
# decoded plasma state and the photon momentum, the importance bias of the scattering
# decision, and the record of everything the tracker needs to evaluate them. The optical
# depth of a step is dτ = ½ (rate_n + rate_{n+1}) (L_unit h/mₑc²) dλ (grmonty `dtauK`).

"Spawn threshold on the weight of a scattered child (gpumonty sphere model: 0)."
const MC_WEIGHT_MIN = 0.0

"""
    FluidFrame()
    StaticFrame()

How the fluid-frame frequency and the pitch angle of an absorption rate are formed.
`FluidFrame`: from the state's four-vectors, `ν = −(p·u) mₑc²/h` and
`cos θ = (p·b)/(√b² (−p·u))` (grmonty `get_bk_angle`). `StaticFrame`: for a fluid at
rest with its field along x¹ in flat coordinates, from the contravariant wave vector,
`ν = k^t mₑc²/h` and `cos θ = k¹/|k|`. The two agree in exact arithmetic on a null
`k` and differ in rounding. [`rate_frame`](@ref) picks the frame of a provider.
"""
struct FluidFrame end

"The rate frame of a static fluid with its field along x¹ in flat coordinates; see [`FluidFrame`](@ref)."
struct StaticFrame end

"""
    rate_frame(provider) -> FluidFrame() | StaticFrame()

The frame of the absorption rate of `provider`: `StaticFrame()` for a `UniformSphere`,
`FluidFrame()` for every other provider.
"""
@inline rate_frame(::PlasmaProvider) = FluidFrame()
@inline rate_frame(::UniformSphere) = StaticFrame()

"""
    absorption_rate(state::PlasmaState, p::SVector{4}) -> Float64

Invariant thermal synchrotron absorption rate `ν α_ν` (Hz cm⁻¹) of the photon of
covariant wave vector `p` (mₑc²) in the plasma `state`, in the [`FluidFrame`](@ref).
Zero for an invalid state, a photon with `p·u ≥ 0` (not future-directed in the
fluid frame) or a field with `b² ≤ 0`.
"""
@inline function absorption_rate(s::PlasmaState, p::SVector{4})
    s.valid || return 0.0
    kdotu = dot(p, s.ucon)
    kdotu >= 0.0 && return 0.0
    nu = -kdotu * ME * CL * CL / HPL
    b_sq = dot(s.bcon, s.bcov)
    b_sq <= 0.0 && return 0.0
    cth = dot(p, s.bcon) / (sqrt(b_sq) * (-kdotu))
    sth = sqrt(max(0.0, 1.0 - cth * cth))
    return nu * alpha_nu(s.n_e, s.theta_e, s.B, nu, sth)
end

"""
    absorption_rate(::StaticFrame, state::PlasmaState, kcon::SVector{4}) -> Float64

`ν α_ν` (Hz cm⁻¹) of the photon of contravariant wave vector `kcon` in the
[`StaticFrame`](@ref): `ν = k^t mₑc²/h`, `cos θ = k¹/|k|`. Zero for an invalid state
or a vanishing spatial `k`.
"""
@inline function absorption_rate(::StaticFrame, s::PlasmaState, kcon::SVector{4})
    s.valid || return 0.0
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ks <= 0.0 && return 0.0
    cth = kcon[2] / ks
    sth = sqrt(max(0.0, 1.0 - cth * cth))
    nu = kcon[1] * ME * CL * CL / HPL
    return nu * alpha_nu(s.n_e, s.theta_e, s.B, nu, sth)
end

"""
    scattering_rate(table::HotcrossTable, state::PlasmaState, p::SVector{4}) -> Float64

Invariant electron-scattering rate `ν n_e σ_h(hν/mₑc², Θₑ)` (Hz cm⁻¹) of the photon of
covariant wave vector `p` in `state`, with `ν = −(p·u) mₑc²/h` and `σ_h` from the
hot-cross `table`. Zero for an invalid state or `p·u ≥ 0`. For a static fluid
(`u = (1, 0, 0, 0)`, flat metric) `−p·u = k^t` exactly, so one method serves both
frames.
"""
@inline function scattering_rate(table::HotcrossTable, s::PlasmaState, p::SVector{4})
    s.valid || return 0.0
    kdotu = dot(p, s.ucon)
    kdotu >= 0.0 && return 0.0
    nu = -kdotu * ME * CL * CL / HPL
    return alpha_inv_scatt(table, nu, s.theta_e, s.n_e)
end

"""
    Bias{T}(b0)
    Bias(sphere::UniformSphere)

Importance bias of the exponential-transform scattering decision,
`b = max(1, b0 (n_sc + 1)²)` after `n_sc` scatterings ([`bias`](@ref)): a constant
`b0 > 1` raises the rate of rare scatterings and the growth with order keeps the
higher orders populated. The sphere form is `b0 = n_e σ_T R`, its Thomson depth.
The estimator is unbiased for any `b ≥ 1`.
"""
struct Bias{T}
    b0::T
end
Bias(s::UniformSphere) = Bias(s.n_e * SIGMA_THOMSON * s.R)

"""
    bias(b::Bias, nscatt) -> Float64

`b0 (nscatt + 1)²` when it exceeds 1, else 1 (also for a NaN product).
"""
@inline function bias(b::Bias, nscatt::Integer)
    v = b.b0 * (Float64(nscatt) + 1.0)^2
    return v > 1.0 ? v : 1.0
end

"""
    Opacities(; params = nothing, table = nothing, bias = Bias(1.0), L_unit = 1.0,
              weight_min = MC_WEIGHT_MIN, max_scatt = 0)
    Opacities(params::TransferParams; kw...)

Everything [`track!`](@ref) needs beyond the control, the spacetime and the provider:
`params`, the parameter set handed to `Matter.sample` (units, electron model, cuts;
not read by analytic providers such as `UniformSphere`); `table`, the `HotcrossTable`
of the scattering rate (unused without scattering); `bias`, the [`Bias`](@ref);
`L_unit`, centimetres per coordinate length (`params.units.L_unit` in the second
form); `weight_min`, the smallest child weight that is spawned; and `max_scatt`, the
scattering order beyond which no child is spawned. `isbits` after `Adapt.adapt`.
"""
struct Opacities{P,H,T}
    params::P
    table::H
    bias::Bias{T}
    L_unit::Float64
    weight_min::Float64
    max_scatt::Int
end
Adapt.@adapt_structure Opacities

Opacities(; params = nothing, table = nothing, bias::Bias = Bias(1.0), L_unit::Real = 1.0,
          weight_min::Real = MC_WEIGHT_MIN, max_scatt::Integer = 0) =
    Opacities(params, table, bias, Float64(L_unit), Float64(weight_min), Int(max_scatt))
Opacities(params::TransferParams; kw...) =
    Opacities(; params = params, L_unit = params.units.L_unit, kw...)
