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

Constant importance bias of the scattering events, `b = max(1, b0 (n_sc + 1)²)` after
`n_sc` scatterings ([`bias`](@ref)): a constant `b0 > 1` raises the rate of rare
scatterings and the growth with order keeps the higher orders populated. The sphere
form is `b0 = n_e σ_T R`, its Thomson depth. [`StateBias`](@ref) is the bias that
follows the plasma and the photon weight. Any `b ≥ 1` leaves the estimator unbiased.
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
    amplification(theta_e) -> Float64

`A(Θₑ) = 1 + 4Θₑ + 16Θₑ²`, the closed form that joins the two limits of the mean energy
gain of one Thomson scattering off a Maxwell–Jüttner population,
`1 + 4Θₑ K₃(1/Θₑ)/K₂(1/Θₑ)` (`1 + 4Θₑ` cold, `16Θₑ²` hot): at most 15 % above it for every
`Θₑ` (157 against 146.85 at `Θₑ = 3`). Any positive function of the state serves
[`StateBias`](@ref); this one costs no Bessel function.
"""
@inline amplification(theta_e) = 1.0 + theta_e * (4.0 + 16.0 * theta_e)

"""
    StateBias(b0, w_ref, b_max, split = 1, order = 1)

Importance bias of the scattering events that follows the plasma and the photon: at a
step of a photon of weight `w`, scattered `n` times, through gas of temperature `Θₑ`,
`b = clamp(b0 A(Θₑ) (w/w_ref) order^n, 1, b_max)` ([`amplification`](@ref)). With
`order = 1` a child, of weight about `w/b`, carries about `w_ref/(b0 A)` photons and the
energy weight `w_ref E/b0` whatever the gas that scattered it, and its own bias is near 1,
so the light scattered two and three times is sampled little; `order > 1` raises the bias
of every later generation by that factor per scattering, which samples the higher orders
and the high-frequency tail they make. `b_max` bounds the events per step. With
`split > 1`, an event whose uncapped bias exceeds `b_max` spawns
`m = min(split, ⌈b_uncapped/b_max⌉)` children of `1/m` of the event weight, each with its
own kick. Any positive `b` chosen from the state before the step's draws, and any split,
leave the estimator unbiased.
"""
struct StateBias
    b0::Float64
    w_ref::Float64
    b_max::Float64
    split::Int
    order::Float64
end
StateBias(b0::Real, w_ref::Real, b_max::Real, split::Integer = 1, order::Real = 1.0) =
    StateBias(Float64(b0), Float64(w_ref), Float64(b_max), Int(split), Float64(order))

"""
    event_bias(b, nscatt, theta_e, w) -> (b::Float64, copies::Int)

The bias of the scattering events of a step and the children per event: for a
[`Bias`](@ref) `(bias(b, nscatt), 1)`; for a [`StateBias`](@ref) the clamped
`b0 A(Θₑ) (w/w_ref) order^nscatt` and its split count. `theta_e` is the temperature of the step (the
larger of its two end samples; 0 for vacuum), `w` the photon weight at the step start.
"""
@inline event_bias(b::Bias, nscatt::Integer, theta_e, w) = (bias(b, nscatt), 1)
@inline function event_bias(b::StateBias, nscatt::Integer, theta_e, w)
    t = b.b0 * amplification(theta_e) * (w / b.w_ref) * b.order^nscatt
    t > 1.0 || return (1.0, 1)
    t < b.b_max && return (t, 1)
    c = t / b.b_max
    m = c >= b.split ? b.split : max(1, unsafe_trunc(Int, ceil(c)))
    return (b.b_max, m)
end

"""
    Opacities(; params = nothing, table = nothing, bias = Bias(1.0), L_unit = 1.0,
              weight_min = MC_WEIGHT_MIN, max_scatt = 0)
    Opacities(params::TransferParams; kw...)

Everything [`track!`](@ref) needs beyond the control, the spacetime and the provider:
`params`, the parameter set handed to `Matter.sample` (units, electron model, cuts;
not read by analytic providers such as `UniformSphere`); `table`, the `HotcrossTable`
of the scattering rate (unused without scattering); `bias`, a [`Bias`](@ref) or a
[`StateBias`](@ref); `L_unit`, centimetres per coordinate length
(`params.units.L_unit` in the second form); `weight_min`, the child weight at or below
which an event spawns nothing; and `max_scatt`, the scattering order beyond which no
child is spawned. `isbits` after `Adapt.adapt`.
"""
struct Opacities{P,H,B}
    params::P
    table::H
    bias::B
    L_unit::Float64
    weight_min::Float64
    max_scatt::Int
end
Adapt.@adapt_structure Opacities

Opacities(; params = nothing, table = nothing, bias = Bias(1.0), L_unit::Real = 1.0,
          weight_min::Real = MC_WEIGHT_MIN, max_scatt::Integer = 0) =
    Opacities(params, table, bias, Float64(L_unit), Float64(weight_min), Int(max_scatt))
Opacities(params::TransferParams; kw...) =
    Opacities(; params = params, L_unit = params.units.L_unit, kw...)
