# fig_deflection.jl — midplane Kerr deflection angles (paper figures 1 and 2): 51 rays across
# the middle row of a plane-parallel camera of width 36 M at r = 1000 M, theta = 90 deg,
# around a spin-0.9 hole, traced until they return to their starting radius, their azimuthal
# deflection compared with the finite-distance formula of Iyer & Hansen 2009 (paper eq.
# deflection, both legs integrated numerically after the substitution u = u0 - w^2). Rays
# with b_- <= b <= b_+ (inside the photon ring) are excluded. Figure 1: the adaptive Vern7
# tracer at reltol = abstol = 1e-8. Figure 2: errors against the number of steps for the
# Vern7 tracer over tolerances 1e-12 .. 1e-5 and for the fixed-control RK4 march with
# step (r - r_h)/C over C = 2 .. 1000 (in place of the paper's DP, RK4 and RK2 ladders).
# Writes out/blacklight/deflection/{values.json, errors.json}.
# Run from the repository root:   julia -t 16 --project=BLP BLP/figures/blacklight/fig_deflection.jl
include(joinpath(@__DIR__, "common.jl"))
using BLP.Spacetime: trace_geodesic, PlaneParallelCamera, CameraFrame, camera_ray, pixel_centres,
                     MarchBudget, march_control, rk4_step, horizon_radius
using LinearAlgebra: SymTridiagonal, eigen

const A = 0.9
const RCAM = 1000.0
const WIDTH = 36.0
const NPIX = 51

# Gauss-Legendre nodes and weights on [-1, 1] (Golub-Welsch)
function gauss_legendre(n)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end
const GL = gauss_legendre(400)

# |dφ/du| of the midplane geodesic with impact parameter b (paper eq. deflection b)
function dphidu(u, b, a)
    M = 1.0
    num = 1 - 2M * (1 - a / b) * u
    den = 1 - 2M * u + a * a * u * u
    rad = 2M * (1 - a / b)^2 * u^3 - (1 - a * a / (b * b)) * u^2 + 1 / (b * b)
    return abs(num / den) / sqrt(max(rad, 0.0))
end

# closest approach u0 = 1/r0 (paper eq. deflection c)
function u_closest(b, a)
    M = 1.0
    q = 1 - a * a / (b * b)
    arg = -3 * sqrt(3.0) * M * (1 - a / b)^2 / (abs(b) * q^1.5)
    r0 = 2abs(b) / sqrt(3.0) * sqrt(q) * cos(acos(clamp(arg, -1.0, 1.0)) / 3)
    return 1 / r0
end

# total deflection from r_cam in to r0 and back out to r_cam: 2 ∫_{u_cam}^{u0} |dφ/du| du,
# with u = u0 - w^2 (dw = -du/(2w)) removing the inverse-square-root endpoint
function deflection_formula(b, a, rcam)
    u0 = u_closest(b, a)
    ucam = 1 / rcam
    wmax = sqrt(u0 - ucam)
    s = 0.0
    for (xg, wg) in zip(GL...)
        w = wmax / 2 * (xg + 1)
        u = u0 - w * w
        s += wg * wmax / 2 * dphidu(u, b, a) * 2w
    end
    return 2s * sign(b)
end

# unwrapped azimuth change along a list of (x, y) points of the backward trace; the
# photon's own change from source to camera is its negative
function unwrapped_dphi(xs, ys)
    φ0 = atan(ys[1], xs[1])
    tot = 0.0; prev = φ0
    for k in 2:length(xs)
        φ = atan(ys[k], xs[k])
        d = φ - prev
        d = d - 2π * round(d / (2π))
        tot += d; prev = φ
    end
    return tot
end

function main()
    out = outdir("deflection")
    st = KerrSchildCartesian(A)
    cam = PlaneParallelCamera(inc_deg = 90.0, d = RCAM, fov = WIDTH, npix = NPIX)
    fr = CameraFrame(cam)
    αs = pixel_centres(cam)
    bp = -A + 6cos(acos(-A) / 3); bm = -A - 6cos(acos(A) / 3)
    rays = []
    for α in αs
        y0, ok = camera_ray(st, fr, α, 0.0)
        ok || error("no null root")
        b = -(y0[2] * y0[7] - y0[3] * y0[6]) / y0[5]          # L_z/E of k = -p
        inside = bm <= b <= bp
        push!(rays, (α = α, y0 = y0, b = b, inside = inside, r0 = sqrt(y0[2]^2 + y0[3]^2 + y0[4]^2)))
    end
    logf("photon ring b- = %.4f, b+ = %.4f; %d of %d rays excluded\n", bm, bp, count(r -> r.inside, rays), NPIX)

    # figure 1: Vern7 at 1e-8
    function vern(ray, tol)
        sol = trace_geodesic(st, ray.y0; escape = ray.r0, reltol = tol, abstol = tol, save_everystep = true, λmax = 1e4)
        xs = [u[2] for u in sol.u]; ys = [u[3] for u in sol.u]
        return -unwrapped_dphi(xs, ys), length(sol.t) - 1
    end
    vals = Dict{String,Any}[]
    for r in rays
        r.inside && continue
        dφ, nst = vern(r, 1e-8)
        f = deflection_formula(r.b, A, r.r0)
        push!(vals, Dict("alpha" => r.α, "b" => r.b, "dphi" => dφ, "formula" => f, "steps" => nst,
                         "flat" => sign(r.b) * π))
        logf("alpha %7.3f  b %8.4f  dphi %10.6f  formula %10.6f  err %.2e  steps %d\n",
             r.α, r.b, dφ, f, dφ - f, nst)
    end
    write_json(joinpath(out, "values.json"), record("deflection_values"; a = A, rcam = RCAM, width = WIDTH,
                        npix = NPIX, b_minus = bm, b_plus = bp, tol = 1e-8, rays = vals))

    # figure 2: error versus steps, Vern7 tolerance ladder and RK4 C ladder
    tols = [10.0^e for e in -12:-5]
    errs = Dict{String,Any}("vern7" => Dict{String,Any}(), "rk4" => Dict{String,Any}())
    for tol in tols
        rows = Dict{String,Any}[]
        for r in rays
            r.inside && continue
            dφ, nst = vern(r, tol)
            push!(rows, Dict("b" => r.b, "err" => abs(dφ - deflection_formula(r.b, A, r.r0)), "steps" => nst))
        end
        errs["vern7"][string(tol)] = rows
        logf("vern7 tol %.0e: median steps %d, median err %.2e\n", tol,
             round(Int, sum(x -> x["steps"], rows) / length(rows)), sort([x["err"] for x in rows])[end ÷ 2 + 1])
    end
    Cs = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    r_h = horizon_radius(st)
    for C in Cs
        rows = Dict{String,Any}[]
        for r in rays
            r.inside && continue
            b = MarchBudget(C, r.r0, Inf, 2_000_000)
            y = r.y0; r_prev = Inf; n = 0
            xs = [y[2]]; ys = [y[3]]; us = [1 / sqrt(y[2]^2 + y[3]^2 + y[4]^2)]
            while true
                m = march_control(st, y, r_prev, b)
                m.stop && break
                r_prev = m.R
                y = rk4_step(st, y, m.h); n += 1
                push!(xs, y[2]); push!(ys, y[3]); push!(us, 1 / sqrt(y[2]^2 + y[3]^2 + y[4]^2))
            end
            # the last step overshoots the camera radius: interpolate φ linearly in u = 1/r
            dφ_all = unwrapped_dphi(xs, ys)
            dφ_last = unwrapped_dphi(xs[end-1:end], ys[end-1:end])
            u_target = 1 / r.r0
            f = (us[end-1] - u_target) / (us[end-1] - us[end])
            dφ = -(dφ_all - dφ_last + f * dφ_last)
            push!(rows, Dict("b" => r.b, "err" => abs(dφ - deflection_formula(r.b, A, r.r0)), "steps" => n))
        end
        errs["rk4"][string(C)] = rows
        logf("rk4 C %.0f: median steps %d, median err %.2e\n", C,
             round(Int, sum(x -> x["steps"], rows) / length(rows)), sort([x["err"] for x in rows])[end ÷ 2 + 1])
    end
    write_json(joinpath(out, "errors.json"), record("deflection_errors"; a = A, rcam = RCAM, tols = tols, Cs = Cs, ladders = errs))
end
main()
