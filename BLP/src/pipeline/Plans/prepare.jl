# `prepare`: every derived quantity of a run, computed once. The camera, the frame, the
# units and the transfer parameters follow the configuration; the horizon, the escape
# and emission radii and the march budget follow the spacetime and the matter; the queue
# order and the rescue set follow the Kerr critical curve.

"""
    transfer_params(cfg::RunConfig; mass_scale = cfg.mass_scale_cgs,
                    rat_high = cfg.plasma.rat_high) -> TransferParams

The march parameter set of `cfg`: `Units(m_bh_cgs, mass_scale)`, the electron model of
`cfg.plasma`, the cuts `sigma ≤ sigma_cut` and `theta_e ≥ theta_e_min` (a negative value
turns a cut off), the power-law population when `power_frac != 0`, the kappa
population when `kappa_frac != 0`, and the observing frequency `frequency_hz`.
`mass_scale` or `rat_high` may be a `ForwardDiff.Dual` (the units or the electron model
then carry the derivative), or another value of the mass scale (one set of a
mass-scale ladder).
"""
function transfer_params(cfg::RunConfig; mass_scale = cfg.mass_scale_cgs,
                         rat_high = cfg.plasma.rat_high)
    p = cfg.plasma
    E = typeof(float(rat_high))
    electrons = ElectronModel{E}(mu = p.mu, ne_ni = p.ne_ni, use_p = p.use_p,
                                 rat_low = p.rat_low, rat_high = rat_high, gamma = p.gamma,
                                 gamma_i = p.gamma_i, gamma_e = p.gamma_e,
                                 thermal_frac = p.thermal_frac)
    cuts = PlasmaCuts(sigma = legacy_bound(Float64, -1.0, p.sigma_cut),
                      theta_e = legacy_bound(Float64, p.theta_e_min, -1.0))
    powerlaw = p.power_frac != 0.0 ? PowerLaw(p.power_frac, p.power_p, p.gamma_min, p.gamma_max) :
               nothing
    kappa = p.kappa_frac != 0.0 ? Kappa(p.kappa_frac, p.kappa, p.w) : nothing
    return TransferParams(Units(cfg.m_bh_cgs, mass_scale), electrons, cuts, cfg.frequency_hz;
                          powerlaw, kappa)
end

"""
    prepare(cfg::RunConfig; spacetime, matter, mode = IQUV(), lanes = Lanes(), tile = 16)
        -> Plan

The plan of a run of `cfg` on `spacetime` (a `Float64` spacetime value: Kerr, a binary,
an ADM grid, a user metric) through `matter` (a plasma provider: a snapshot, a uniform
grid, a frame window, an analytic source; or `nothing` for products that need no
plasma). `mode` is the transfer mode of the imaging products, `lanes` the precision
lanes ([`Lanes`](@ref), [`MixedLanes`](@ref)), `tile` the side of the image tiles of the
Kerr queue order.

Derived here, once: the camera and its frame, the pixel centres, the transfer
parameters ([`transfer_params`](@ref)), the escape radius `escape_factor × distance`,
the emission radius `R_em = domain_radius(matter)` beyond which the march stops on an
outbound ray, the march budget on both lanes, the horizon proxy, the observer
inclination, the Kerr critical curve with the cost-ordered queue (identity order on
other spacetimes) and, for `MixedLanes`, the rescue set. Throws `ArgumentError` for a
spacetime that is not `Float64`, a geodesic lane the spacetime does not have, transfer
or accumulator lanes other than `Float64`, or `MixedLanes` off Kerr.
"""
function prepare(cfg::RunConfig; spacetime, matter, mode::TransferMode = IQUV(),
                 lanes::Union{Lanes,MixedLanes} = Lanes(), tile::Integer = 16)
    spacetime isa Spacetime{Float64} ||
        throw(ArgumentError("prepare: the spacetime must have Float64 scalars; got " *
                            "$(typeof(spacetime))"))
    return plan_of(cfg, spacetime, matter, mode, lanes, Int(tile), transfer_params(cfg))
end

"""
    prepare(sphere::UniformSphere) -> Plan

The plan of the Monte Carlo sphere: the flat spacetime `Minkowski()` with `sphere` as its
matter, in centimetres. The coordinates are cm and the parameter set is `nothing`: the
sphere's plasma is given in cgs and reads no units, electron model or cuts, and the
Monte Carlo opacities of a plan without a parameter set take one centimetre per
coordinate length. The configuration is `RunConfig()`; its camera, march budget and
queue are not read by the Monte Carlo, and a plan without a parameter set is not imaged.
"""
prepare(sphere::UniformSphere) =
    plan_of(RunConfig(), Minkowski(), sphere, StokesI(), Lanes(), 16, nothing)

# The plan of `cfg` on `spacetime` through `matter` with the parameter set `params`.
function plan_of(cfg::RunConfig, spacetime, matter, mode, lanes, tile::Int, params)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    escape = cfg.escape_factor * cfg.camera.distance
    R_em = matter === nothing ? Inf : Float64(domain_radius(matter))
    budget = MarchBudget{Float64}(cfg.C, escape, R_em, cfg.max_steps)
    stg, bg = geodesic_lane(spacetime, budget, lanes)
    curve = scheduling_curve(spacetime, cam)
    queue = isempty(curve[1]) ? collect(1:npixels(cam)) :
            cost_order(curve, cam.fov, cam.npix; tile = tile)
    return Plan(cfg, spacetime, stg, matter, mode, params, cam, CameraFrame(cam),
                PixelGrid(cam.fov, cam.npix), budget, bg, horizon_of(spacetime), escape, R_em,
                deg2rad(cfg.camera.inclination_deg), curve, lanes, queue,
                rescue_set(spacetime, curve, cam, lanes),
                Ref{Union{Nothing,HotcrossTable{Matrix{Float64}}}}(nothing))
end

# The geodesic-lane spacetime and budget of `lanes`.
function geodesic_lane(st, budget, ::Lanes{Tg,Tt,A}) where {Tg,Tt,A}
    (Tt === Float64 && A === Float64) ||
        throw(ArgumentError("Lanes{$Tg,$Tt,$A}: the marches run the transfer and the " *
                            "accumulator in Float64"))
    return lane_pair(st, budget, Tg)
end
geodesic_lane(st, budget, ::MixedLanes) = lane_pair(st, budget, Float32)

function lane_pair(st, budget, ::Type{T}) where {T}
    stg = lane(st, T)
    stg isa Spacetime{T} ||
        throw(ArgumentError("$(nameof(typeof(st))) has no $T geodesic lane"))
    return stg, lane(budget, T)
end

# The Kerr critical curve at the scheduling inclination; none on other spacetimes.
scheduling_curve(st::KerrSchildCartesian, cam) =
    critical_curve(st.a, curve_inclination(cam.inc_deg))
scheduling_curve(st, cam) = (Float64[], Float64[])

rescue_set(st, curve, cam, ::Lanes) = Int[]
function rescue_set(st, curve, cam, lanes::MixedLanes)
    st isa KerrSchildCartesian ||
        throw(ArgumentError("MixedLanes: the rescue set is the neighbourhood of the Kerr " *
                            "critical curve; $(nameof(typeof(st))) has none"))
    return rescue_pixels(curve, cam.fov, cam.npix; rescue_px = lanes.rescue_px)
end

horizon_of(st) = applicable(horizon_radius, st) ? horizon_radius(st) : NaN
