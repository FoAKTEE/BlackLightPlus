# The order key of a Monte Carlo photon: the radial shell of its position above the
# horizon floor, the octant of its position and the octant of its spatial wave vector.
# Results never depend on the order in which photons are visited (children are stored by
# atomic arrival and every reader is order-free), so the key only groups photons whose
# plasma reads and step counts are alike.

"Width in `log r` of a radial shell of [`photon_key`](@ref)."
const KEY_SHELL_WIDTH = 0.05
"Number of radial shells of [`photon_key`](@ref); the last one holds every larger radius."
const KEY_SHELLS = 128
"Number of distinct values of [`photon_key`](@ref): `KEY_SHELLS` shells × 8 × 8 octants."
const PHOTON_KEYS = KEY_SHELLS * 64
"Ratio of the shell floor to the march radius when the spacetime has no horizon floor."
const KEY_FLOOR_RATIO = exp(-KEY_SHELLS * KEY_SHELL_WIDTH)

"""
    photon_key(st, ctl, y::SVector{8,Float64}) -> Int

The order key `(s × 8 + octant(x)) × 8 + octant(k)` in `0:8191` of the state `y` (position
`x`, covariant momentum `p`) on the spacetime `st` under the Monte Carlo march control
`ctl` ([`FlatMarch`](@ref) or [`CurvedMarch`](@ref)). A total function:

- `s` is the shell of the coordinate radius `r = √(x² + y² + z²)` above the floor
  `r_min`: `horizon_floor(st)` when that is positive and finite, otherwise the march
  radius of `ctl` (the `R` of a `FlatMarch`, `min(R_em, escape)` of a `CurvedMarch`)
  times `e^−6.4`. `s = 0` when `r` is not finite or `r ≤ r_min` (or `r_min` is not
  positive), else `unsafe_trunc(Int, min((log r − log r_min) / 0.05, 127.0))`;
- `octant(v) = 4 signbit(v¹) + 2 signbit(v²) + signbit(v³)` of the position and of the
  spatial wave vector `k^i = g^{iν} p_ν`, `g^{μν}` the inverse metric of `st` at `x`;
- a state with a non-finite entry of `x`, `p` or `k` has key 0.

The logarithm is `tang_log` and `r` and `k` are sums of plain products, so the shell and
the position octant of a state are the same on the host and on a device; the direction
octant follows each backend's rounding of the inverse metric. The key orders photons and
enters no result.
"""
@inline function photon_key(st, ctl, y::SVector{8,Float64})
    all(isfinite, y) || return 0
    x = position(y)
    p = momentum(y)
    g = inverse_metric(st, x)
    k1 = ((g[2, 1] * p[1] + g[2, 2] * p[2]) + g[2, 3] * p[3]) + g[2, 4] * p[4]
    k2 = ((g[3, 1] * p[1] + g[3, 2] * p[2]) + g[3, 3] * p[3]) + g[3, 4] * p[4]
    k3 = ((g[4, 1] * p[1] + g[4, 2] * p[2]) + g[4, 3] * p[3]) + g[4, 4] * p[4]
    (isfinite(k1) && isfinite(k2) && isfinite(k3)) || return 0
    r = sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4])
    r_min = key_floor(st, ctl)
    s = isfinite(r) && 0.0 < r_min < r ?
        unsafe_trunc(Int, min((tang_log(r) - tang_log(r_min)) / KEY_SHELL_WIDTH, KEY_SHELLS - 1.0)) : 0
    return (s * 8 + octant(x[2], x[3], x[4])) * 8 + octant(k1, k2, k3)
end

"The sign bits of a spatial vector as an octant in `0:7`, the first component the highest bit."
@inline octant(a::Float64, b::Float64, c::Float64) = 4 * signbit(a) + 2 * signbit(b) + signbit(c)

"The floor `r_min` of the shells of [`photon_key`](@ref) on `st` under `ctl`."
@inline function key_floor(st, ctl)
    r_h = Float64(horizon_floor(st))
    return 0.0 < r_h < Inf ? r_h : march_radius(ctl) * KEY_FLOOR_RATIO
end

"The radius past which the march `ctl` records its photon: the ball of a `FlatMarch`, `min(R_em, escape)` of a `CurvedMarch`."
@inline march_radius(c::FlatMarch) = Float64(c.R)
@inline march_radius(c::CurvedMarch) = Float64(min(c.R_em, c.escape))
