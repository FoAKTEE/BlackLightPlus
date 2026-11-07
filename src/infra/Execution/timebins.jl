# Angular and arrival-time rules of the exact bins: inclination bands with azimuth sectors,
# and any rule repeated over uniform bins of the arrival time at a distant observer, with
# an underflow and an overflow bin, for light curves and time-resolved spectra. Both are
# `Bands`, so an `ExactBins` of either is accumulated, merged, gathered and rounded like
# every other.

"""
    SectorBands(incl_deg, half_width_deg, n_az)
    SectorBands(bands::InclinationBands, n_az)

Inclination bands split into `n_az` azimuth sectors: rows `1:N` and `N + 1` are those of
`InclinationBands(incl_deg, half_width_deg)` (the `N` bands and the all-sky row), and row
[`sector_row`](@ref)`(s, k, j)` holds the photons of band `k` whose escape direction `d`
has the azimuth `φ = atan(d², d¹) ∈ [0, 2π)` in `[(j − 1)Δφ, jΔφ)`, `Δφ = 2π/n_az`, of
solid angle `ΔΩ_k/n_az`. The band and all-sky rows are bitwise those of the inclination
bands alone. Fields: `bands`, `n_az`, `dphi`.
"""
struct SectorBands{N} <: Bands
    bands::InclinationBands{N}
    n_az::Int
    dphi::Float64
end
function SectorBands(b::InclinationBands, n_az::Integer)
    n_az >= 1 || throw(ArgumentError("SectorBands: n_az must be positive"))
    return SectorBands(b, Int(n_az), 2.0 * π / n_az)
end
SectorBands(incl_deg, half_width_deg::Real, n_az::Integer) =
    SectorBands(InclinationBands(incl_deg, half_width_deg), n_az)

nrows(s::SectorBands{N}) where {N} = N + 1 + N * s.n_az
function solid_angles(s::SectorBands{N}) where {N}
    dO = solid_angles(s.bands)
    return [dO; [dO[k] / s.n_az for k in 1:N for _ in 1:s.n_az]]
end
total_row(::SectorBands{N}) where {N} = N + 1

"""
    sector_row(s::SectorBands, k, j) -> Int

The row of azimuth sector `j ∈ 1:n_az` of inclination band `k`.
"""
sector_row(s::SectorBands{N}, k::Integer, j::Integer) where {N} = N + 1 + (k - 1) * s.n_az + j

@inline function record_rows!(b::ExactBins, r::SectorBands{N}, off::Int, ie::Int, E, w,
                              d) where {N}
    ksp = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    ksp <= 0.0 && return nothing
    th = acos(clamp(d[3] / ksp, -1.0, 1.0))
    phi = atan(d[2], d[1])
    phi < 0.0 && (phi += 2.0 * π)
    j = clamp(unsafe_trunc(Int, floor(phi / r.dphi)) + 1, 1, r.n_az)
    add_cell!(b, off + N + 1, ie, E, w)
    for k in 1:N
        if th >= r.bands.lo[k] && th <= r.bands.hi[k]
            add_cell!(b, off + k, ie, E, w)
            add_cell!(b, off + N + 1 + (k - 1) * r.n_az + j, ie, E, w)
        end
    end
    return nothing
end

"""
    TimedBands(bands::Bands, t_lo, t_hi, n_t)

The angular rule `bands` repeated over arrival-time bins: `n_t` equal bins of width
`dt = (t_hi − t_lo)/n_t` over `[t_lo, t_hi)`, after an underflow bin (`t < t_lo`) and
before an overflow bin (`t ≥ t_hi`), so every recorded photon has a bin. Arrival bin
`k ∈ 0:n_t + 1` ([`arrival_bin`](@ref)) holds the rows
`k · nrows(bands) .+ (1:nrows(bands))` of an `ExactBins`, filled by
`record!(bins, E, w, dir, t)`; summed over `k` they are the bins of `bands` alone. Fields:
`bands`, `t_lo`, `t_hi`, `dt`, `n_t`.
"""
struct TimedBands{B<:Bands} <: Bands
    bands::B
    t_lo::Float64
    t_hi::Float64
    dt::Float64
    n_t::Int
end
function TimedBands(bands::Bands, t_lo::Real, t_hi::Real, n_t::Integer)
    (n_t >= 1 && t_hi > t_lo && isfinite(t_lo) && isfinite(t_hi)) ||
        throw(ArgumentError("TimedBands: need n_t ≥ 1 and finite t_lo < t_hi"))
    return TimedBands(bands, Float64(t_lo), Float64(t_hi), (Float64(t_hi) - Float64(t_lo)) / n_t,
                      Int(n_t))
end

nrows(r::TimedBands) = (r.n_t + 2) * nrows(r.bands)
solid_angles(r::TimedBands) = repeat(solid_angles(r.bands), r.n_t + 2)
total_row(::TimedBands) = 0

"""
    arrival_bin(r::TimedBands, t) -> Int

The arrival bin of time `t`: `0` below `t_lo` (and for a NaN), `k` for
`t_lo + (k − 1) dt ≤ t < t_lo + k dt`, `n_t + 1` from `t_hi` on.
"""
@inline function arrival_bin(r::TimedBands, t::Float64)
    v = floor((t - r.t_lo) / r.dt)
    v >= 0.0 || return 0
    return v < r.n_t ? unsafe_trunc(Int, v) + 1 : r.n_t + 1
end

"""
    record!(bins::ExactBins{<:TimedBands}, E, w, dir::SVector{3}, t) -> nothing

Add an escaped photon of energy `E`, weight `w`, escape direction `dir` and arrival time
`t` into its frequency bin of every row of its arrival bin that its direction selects.
Atomic; device-safe.
"""
@inline function record!(b::ExactBins{<:TimedBands}, E::Float64, w::Float64,
                         dir::SVector{3,Float64}, t::Float64)
    ie = energy_bin(b, E)
    ie == 0 && return nothing
    r = b.bands
    record_rows!(b, r.bands, arrival_bin(r, t) * nrows(r.bands), ie, E, w, dir)
    return nothing
end
