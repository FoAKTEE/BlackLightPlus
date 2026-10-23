# Parameter derivatives and parameter ladders of the Stokes I image. The geometry does not
# depend on the parameters, so the geodesic march is shared and only the transfer carries
# the derivative (a `Dual` parameter set) or the ladder (a tuple of parameter sets).

"Tag of the derivative carried by [`sensitivity`](@ref)."
struct SensitivityTag end

"""
    sensitivity(plan, param::Symbol; backend = nothing) -> NamedTuple

The Stokes I image of `plan` and its exact derivative with respect to `param`, by
forward-mode differentiation of the transfer (the per-step transfer is affine in the
intensity, so the derivative of the march is exact): `param = :mass_scale` seeds the
density mass scale (cgs, in the units), `param = :rat_high` the β → ∞ ion-to-electron
temperature ratio (in the electron model); anything else is an `ArgumentError`. Returns
`(image, derivative, flux, dflux, param)`: `image` an [`Image`](@ref) (Stokes I),
`derivative` the `npix × npix` map `∂I/∂param`, `flux` the sum of the image over the
pixels in camera order and `dflux` its derivative. Runs on the plan's lanes, on the
executor `resolve(backend)`.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function sensitivity(plan::Plan, param::Symbol; backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    params = seeded_params(plan.config, param)
    out = Vector{output_type(StokesI(), params)}(undef, npixels(plan))
    march_frame!(exec, plan, out; mode = StokesI(), params)
    n = plan.camera.npix
    flux = sum(out)
    image = Image(plan, value.(out), run_record(exec, plan, t0))
    return (image = image, derivative = reshape(partials.(out, 1), n, n),
            flux = value(flux), dflux = partials(flux, 1), param = param)
end

# The parameter set of `cfg` with `param` seeded as a Dual of unit partial.
function seeded_params(cfg, param::Symbol)
    param === :mass_scale &&
        return transfer_params(cfg; mass_scale = Dual{SensitivityTag}(cfg.mass_scale_cgs, 1.0))
    param === :rat_high &&
        return transfer_params(cfg; rat_high = Dual{SensitivityTag}(cfg.plasma.rat_high, 1.0))
    throw(ArgumentError("sensitivity: param must be :mass_scale or :rat_high, got :$param"))
end

"""
    variants(plan, mass_scales; backend = nothing) -> NamedTuple

`K = length(mass_scales)` Stokes I images of `plan` in one pass, image `k` for the
density mass scale `mass_scales[k]` (g): each ray is marched and its plasma sampled once,
and the transfer runs once per mass scale on the shared samples (the parameter sets
differ only in their units). Image `k` equals the image of the plan prepared with
`mass_scale_cgs = mass_scales[k]`. Grid providers only (the samples are code-unit
primitives); anything else is an `ArgumentError`. Returns `(images, mass_scales, plan,
run)`, `images` a vector of `npix × npix` matrices.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function variants(plan::Plan, mass_scales::AbstractVector{<:Real}; backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    plan.provider isa GridProvider ||
        throw(ArgumentError("variants: the mass-scale ladder samples code-unit primitives " *
                            "and needs a grid provider; got $(typeof(plan.provider))"))
    K = length(mass_scales)
    K >= 1 || throw(ArgumentError("variants: mass_scales must be nonempty"))
    params = ntuple(k -> transfer_params(plan.config; mass_scale = Float64(mass_scales[k])), K)
    out = Vector{SVector{K,Float64}}(undef, npixels(plan))
    march_frame!(exec, plan, out; mode = StokesI(), params)
    n = plan.camera.npix
    return (images = [reshape(getindex.(out, k), n, n) for k in 1:K],
            mass_scales = collect(Float64, mass_scales), plan = plan,
            run = run_record(exec, plan, t0))
end
