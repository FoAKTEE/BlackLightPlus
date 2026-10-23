# The plan: everything a product needs, derived once from a configuration, a spacetime
# and a matter source. Immutable and backend-free: the same plan runs on any executor.
# The one value computed later is the Monte Carlo hot-cross table, memoised on first use.

"""
    Plan

The prepared run that every product consumes; built by [`prepare`](@ref), never by
hand. Fields:

- `config::RunConfig`, the configuration it was prepared from;
- `spacetime`, the `Float64` spacetime of the transfer lane (the plasma decode, the
  wave vector, the tetrads and the camera rays);
- `geodesic_spacetime`, the same spacetime on the geodesic lane (`lane(spacetime, Tg)`;
  the `Float32` bulk lane under `MixedLanes`);
- `provider`, the matter (a `PlasmaProvider`, or `nothing` for products without
  plasma);
- `mode`, the transfer mode (`StokesI()` or `IQUV()`);
- `params::TransferParams`, units, electron model, cuts, nonthermal populations and
  frequency (`nothing` for the Monte Carlo sphere, whose plasma is cgs and whose
  coordinates are centimetres);
- `camera::PlaneParallelCamera`, `frame::CameraFrame{Float64}` and `pixels`, the
  `PixelGrid` of the camera's pixel centres;
- `budget::MarchBudget{Float64}` (`C`, `escape`, `R_em`, `max_steps`) and
  `geodesic_budget`, the same budget on the geodesic lane;
- `horizon`, the spacetime's horizon proxy (`horizon_radius(spacetime)`: the Kerr outer
  horizon, the per-hole radii of a binary, the lapse floor of an ADM grid; `NaN` when
  the spacetime has none);
- `escape = escape_factor × distance` and `R_em = domain_radius(provider)` (`Inf`
  without matter), in M;
- `theta_o`, the observer inclination in radians;
- `curve`, the Kerr critical curve `(α, β)` at the scheduling inclination
  (`curve_inclination`), empty vectors on other spacetimes;
- `lanes`, the precision lanes; `queue`, the pixel queue order (a permutation of
  `1:npixels`: the `queue` given to `prepare`, else longest predicted march first on
  Kerr and the identity elsewhere); `rescue`, the ascending pixels
  marched on the `Float64` lane under `MixedLanes` (empty otherwise);
- `hotcross`, the memo of [`hotcross_table`](@ref): empty until a Monte Carlo
  scattering run of the plan first needs the table.
"""
struct Plan{S,Sg,P,M,T,Bg,H,L}
    config::RunConfig
    spacetime::S
    geodesic_spacetime::Sg
    provider::P
    mode::M
    params::T
    camera::PlaneParallelCamera
    frame::CameraFrame{Float64}
    pixels::PixelGrid{Vector{Float64}}
    budget::MarchBudget{Float64}
    geodesic_budget::Bg
    horizon::H
    escape::Float64
    R_em::Float64
    theta_o::Float64
    curve::Tuple{Vector{Float64},Vector{Float64}}
    lanes::L
    queue::Vector{Int}
    rescue::Vector{Int}
    hotcross::Base.RefValue{Union{Nothing,HotcrossTable{Matrix{Float64}}}}
end

"""
    hotcross_table(plan::Plan) -> HotcrossTable

The hot-cross-section table of the plan's Monte Carlo scattering, `build_hotcross_table()`,
built at the first request and kept in the plan, so later runs of the plan reuse the
same table (and its device copies). Concurrent first requests may each build it; the
tables are identical.
"""
function hotcross_table(plan::Plan)
    t = plan.hotcross[]
    t === nothing || return t
    t = build_hotcross_table()
    plan.hotcross[] = t
    return t
end

"""
    npixels(plan::Plan) -> Int

Pixels of the plan's camera, `resolution²`.
"""
npixels(plan::Plan) = npixels(plan.camera)

"""
    output_type(plan::Plan) -> Type

Per-pixel value type of the plan's imaging march: `Float64` for `StokesI`,
`SVector{4,Float64}` for `IQUV`. A plan without a parameter set (the Monte Carlo
sphere's) has none: `ArgumentError`.
"""
function output_type(plan::Plan)
    plan.params === nothing &&
        throw(ArgumentError("imaging needs a parameter set: prepare the plan from a RunConfig"))
    return output_type(plan.mode, plan.params)
end

function Base.show(io::IO, plan::Plan)
    c = plan.camera
    print(io, "Plan(", nameof(typeof(plan.spacetime)), ", ",
          plan.provider === nothing ? "no matter" : nameof(typeof(plan.provider)), ", ",
          nameof(typeof(plan.mode)), ", ", plan.lanes, ", ", c.npix, "×", c.npix, " pixels)")
end
