# fig_dyn_rays.jl — rays through an orbiting binary in the time-dependent superposed
# Kerr-Schild metric (`BoostedSuperposedKerrSchild` on a `CircularOrbit`), traced backward
# from the camera with ∂_t g kept, for the two figures drawn by plot_dyn_rays.py:
#   fan   — a pinhole fan of NFAN rays in the orbital plane from a camera at (XCAM, 0, 0),
#           half-angle HALF, animated in coordinate time (the ray-fan animation);
#   bohn  — NBOHN plane-parallel rays on the image row β = 0 of an edge-on camera at
#           distance DCAM, spanning the far hole's lensed image beyond the near hole's
#           shadow and the next eyebrow (three rays inside it) when the holes are aligned
#           with the line of sight (figure 8 of arXiv:1410.7775), with four snapshots in
#           time; the row is first scanned with NSCAN rays over |α| ≤ ASCAN.
# Binary: masses q/(q+1) and 1/(q+1) (mass ratio Q, total 1), no spin, separation SEP,
# Keplerian circular orbit prograde
# about +z, phase chosen so that the holes are aligned along x (hole 1 nearer the cameras)
# at t = -TALIGN, when the Bohn-row rays pass them. The observer time is t = 0 at the
# camera; the rays run to earlier times and the holes move meanwhile.
# Each ray: adaptive Vern7, reltol 1e-10, abstol 1e-12, λ ≤ LMAX, captured at
# r'_A < r_h,A + 1e-3 (rest-frame radius of hole A) or normal-frame energy > 100, escaped
# at R > RESC outbound; its dense solution is sampled every DLAM in λ. The same rays are
# traced in the unboosted moving superposition (`boost = false`) for the comparison panel.
# Fates: 1 or 2 captured by that hole (the nearer one at the stop), 0 escaped, 3 budget.
#
# Writes BLP/figures/out/dyn_rays/ (Q = 1) or dyn_rays_q<Q>/: for each (set, variant) a
# `<set>_<variant>.npy` with
# rows (ray, λ, t, x, y, z) and `setup.json` (parameters, per-ray fates and image
# coordinates, snapshot times, code SHA, wall times).
# Run from the repository root:   julia -t 16 --project=BLP BLP/figures/fig_dyn_rays.jl
# Env: BLP_DYN_Q (1, the mass ratio), BLP_DYN_SEP (8), BLP_DYN_XCAM (45), BLP_DYN_HALF (20),
#      BLP_DYN_DCAM (60), BLP_DYN_TALIGN (DCAM + 9). The Bohn row is skipped when the scan
#      finds no far-hole captures at α ≥ 0.
using Dates, LinearAlgebra, Printf, StaticArrays
using BLP
using BLP.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit, trace_geodesic, nullify_covariant,
                     null_error, ks_radius, horizon_radius, PlaneParallelCamera, CameraFrame,
                     camera_ray, normal_frame_energy, CAPTURE_REDSHIFT
using BLP.Data: write_npy

const Q = parse(Float64, get(ENV, "BLP_DYN_Q", "1"))
const OUT = joinpath(@__DIR__, "out", Q == 1 ? "dyn_rays" : "dyn_rays_q$(round(Int, Q))")
const ROOT = dirname(dirname(@__DIR__))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- BLP/src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

const MASSES = (Q / (Q + 1), 1 / (Q + 1))
const SEP = parse(Float64, get(ENV, "BLP_DYN_SEP", "8"))
const XCAM = parse(Float64, get(ENV, "BLP_DYN_XCAM", "45"))
const HALF = parse(Float64, get(ENV, "BLP_DYN_HALF", "20"))      # degrees
const DCAM = parse(Float64, get(ENV, "BLP_DYN_DCAM", "60"))
const TALIGN = parse(Float64, get(ENV, "BLP_DYN_TALIGN", string(DCAM + 9)))
const NFAN = 61
const NSCAN = 321
const ASCAN = 8.0
const NBOHN = 16
const LMAX = 400.0
const RESC = 200.0
const DLAM = 0.05
const RELTOL = 1e-10
const ABSTOL = 1e-12

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

# the orbit: aligned along x (hole 1 at +x, nearer the cameras) at t = -TALIGN
const OMEGA = sqrt(sum(MASSES) / SEP^3)
const PHASE = OMEGA * TALIGN
const ORBIT = CircularOrbit(MASSES, SEP; phase = PHASE)
spacetime(boost) = BoostedSuperposedKerrSchild(collect(MASSES), [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                                               ORBIT; boost)

# backward ray from x0 for the received spatial covector direction ks (p_t = +1)
function ray_from(st, x0, ks)
    k = nullify_covariant(st, x0, ks)
    return vcat(x0, k / k[1])
end

# fate of an end state: 1/2 captured (the horizon epsilon or the redshift cut, which in the
# binary ends the approach at r'_A - r_h,A ≈ 0.2) by the nearer hole, 0 escaped, 3 budget
function fate(st, u)
    x = SVector(u[1], u[2], u[3], u[4])
    R = sqrt(u[2]^2 + u[3]^2 + u[4]^2)
    dA = (ks_radius(st, x, 1) - horizon_radius(st, 1), ks_radius(st, x, 2) - horizon_radius(st, 2))
    if minimum(dA) < 2e-3 || normal_frame_energy(st, u) > 0.99CAPTURE_REDSHIFT
        return dA[1] <= dA[2] ? 1 : 2
    end
    R > 0.99RESC && return 0
    return 3
end

# trace y0 and sample its dense solution every DLAM: (path (n×5: λ, t, x, y, z), fate)
function trace_sampled(st, y0)
    sol = trace_geodesic(st, y0; λmax = LMAX, escape = RESC, reltol = RELTOL, abstol = ABSTOL,
                         save_everystep = true)
    λs = collect(0.0:DLAM:sol.t[end])
    λs[end] == sol.t[end] || push!(λs, sol.t[end])
    path = Matrix{Float64}(undef, length(λs), 5)
    for (i, λ) in enumerate(λs)
        u = sol(λ)
        path[i, :] .= (λ, u[1], u[2], u[3], u[4])
    end
    return path, fate(st, sol.u[end])
end

function trace_set(st, rays, name)
    t0 = time()
    paths = Vector{Matrix{Float64}}(undef, length(rays))
    fates = Vector{Int}(undef, length(rays))
    Threads.@threads for i in eachindex(rays)
        paths[i], fates[i] = trace_sampled(st, rays[i])
    end
    logf("%-14s %4d rays  fates 0/1/2/3 = %d/%d/%d/%d  %.1f s\n", name, length(rays),
         count(==(0), fates), count(==(1), fates), count(==(2), fates), count(==(3), fates), time() - t0)
    return paths, fates
end

function write_set(name, paths)
    rows = sum(size(p, 1) for p in paths)
    a = Matrix{Float64}(undef, rows, 6)
    r = 0
    for (i, p) in enumerate(paths)
        n = size(p, 1)
        a[r+1:r+n, 1] .= i - 1
        a[r+1:r+n, 2:6] .= p
        r += n
    end
    write_npy(joinpath(OUT, name * ".npy"), a)
end

setup = Dict{String,Any}(
    "q" => Q, "masses" => collect(MASSES), "separation" => SEP, "omega" => OMEGA, "phase" => PHASE,
    "orbit_radii" => collect(ORBIT.r), "r_h" => [horizon_radius(spacetime(true), A) for A in 1:2],
    "xcam" => XCAM, "half_deg" => HALF, "dcam" => DCAM, "talign" => TALIGN, "nfan" => NFAN, "nscan" => NSCAN,
    "ascan" => ASCAN, "nbohn" => NBOHN, "lmax" => LMAX, "resc" => RESC, "dlam" => DLAM,
    "reltol" => RELTOL, "abstol" => ABSTOL, "code" => GIT_SHA, "threads" => Threads.nthreads(),
    "date" => string(now()))

for boost in (true, false)
    st = spacetime(boost)
    variant = boost ? "boosted" : "unboosted"
    logf("== %s moving superposition\n", variant)

    # (a) the pinhole fan: received photons arrive at (XCAM, 0, 0) from directions θ about -x
    θs = range(-deg2rad(HALF), deg2rad(HALF); length = NFAN)
    x0 = SVector(0.0, XCAM, 0.0, 0.0)
    fan = [ray_from(st, x0, SVector(cos(θ), -sin(θ), 0.0)) for θ in θs]   # k_i ∝ -(direction away)
    paths, fates = trace_set(st, fan, "fan $variant")
    write_set("fan_$variant", paths)
    setup["fan_theta_deg"] = collect(rad2deg.(θs))
    setup["fan_fates_$variant"] = fates

    # (b) the Bohn row: edge-on plane-parallel camera at +x, image row β = 0
    cam = PlaneParallelCamera(inc_deg = 90.0, d = DCAM, fov = 2ASCAN, npix = NSCAN)
    fr = CameraFrame(cam)
    αs = collect(range(-ASCAN, ASCAN; length = NSCAN))
    scan = [first(camera_ray(st, fr, α, 0.0)) for α in αs]
    _, sfates = trace_set(st, scan, "scan $variant")
    setup["scan_alpha"] = αs
    setup["scan_fates_$variant"] = sfates
    if boost
        # runs of equal fate along the row at α ≥ 0; the window spans the first far-hole (2)
        # run beyond the near hole's shadow (the ring) and the next capture run (the
        # eyebrow), with margins, at most 4 M wide
        runs = Tuple{Int,Int,Int}[]                      # (fate, first, last)
        i = findfirst(>=(0.0), αs)
        while i <= NSCAN
            j = i
            while j < NSCAN && sfates[j+1] == sfates[i]; j += 1; end
            push!(runs, (sfates[i], i, j)); i = j + 1
        end
        setup["scan_runs"] = [[f, αs[a], αs[b]] for (f, a, b) in runs]
        k1 = findfirst(r -> r[1] == 2, runs)
        if k1 === nothing
            logf("no far-hole captures on the row at α ≥ 0: the Bohn row is skipped\n")
            global BOHN_ALPHAS = Float64[]
        end
        k2 = k1 === nothing ? nothing : findnext(r -> r[1] in (1, 2), runs, k1 + 1)
        lo = k1 === nothing ? NaN : αs[runs[k1][2]] - 0.2
        if k1 === nothing
        elseif k2 === nothing
            global BOHN_ALPHAS = collect(range(lo, min(αs[runs[k1][3]] + 0.3, lo + 4.0); length = NBOHN))
        else
            # NBOHN - 3 rays up to the eyebrow, 3 inside it (its run may be narrower than
            # the scan step), the rest of the window up to 4 M wide
            e_lo, e_hi = αs[runs[k2][2]], αs[runs[k2][3]]
            hi = min(e_lo - 0.05, lo + 4.0)
            global BOHN_ALPHAS = vcat(collect(range(lo, hi; length = NBOHN - 3)),
                                      collect(range(e_lo, e_hi; length = 3)))
        end
    end
    isempty(BOHN_ALPHAS) && continue
    bohn = [first(camera_ray(st, fr, α, 0.0)) for α in BOHN_ALPHAS]
    paths, bfates = trace_set(st, bohn, "bohn $variant")
    write_set("bohn_$variant", paths)
    setup["bohn_alpha"] = BOHN_ALPHAS
    setup["bohn_fates_$variant"] = bfates
end
setup["fan_frames"] = collect(0.0:-1.0:-99.0)
json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"
open(io -> println(io, json_str(setup)), joinpath(OUT, "setup.json"), "w")
logf("wrote %s\n", OUT)
