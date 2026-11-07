# fig_cmp_disk.jl — BlackLightPlus side of the analytic thin-disk code comparison:
# the closed-form (exact-Kerr) tracer, the RK4 march and the precise march on identical
# screen points.
#
#   closed form: render_analytic on the camera whose pixel centres are AART's grid nodes,
#                orders n = 0, 1, 2, observer at distance D (at infinity to rounding, as
#                AART and Krang) and at D_fin (AART's default distance); the same landings
#                on the radial cuts (pixel_prep, band_landing, intensity).
#   RK4 march:   the fixed-control march of the imaging products (march_control and
#                rk4_step on Kerr in Cartesian Kerr-Schild coordinates, camera at
#                distance D_rk, step (r − r_h)/C), with the disk set as a boundary condition
#                at every equatorial crossing: the crossing is the root in the step length
#                of z(rk4_step(y_k, h)) inside the step that changes the sign of z, the
#                k-th crossing is order n = k − 1, and the redshift is g = 1/(p_μ u^μ) from
#                the marched covector p (p_t = 1) and the disk four-velocity u.
#                C = 80 (default) on the grid and the cuts; C = 20 … 320 on a subset.
#   precise:     render_crossings under Plans.precise (Verner 9 at C = 20 with the winding
#                step, compensated state update, crossings at full order, camera at
#                infinity, redshift cut 1e6, no horizon backstop) on the camera whose pixel
#                centres are AART's grid nodes and on the radial cuts; the k-th crossing of
#                the equator is order n = k − 1, as for the RK4 march.
#
# Writes figures/paper/out/cmp_disk/<config>/ (arrays [iβ, iα]; cuts flattened angle-major):
#   nodes.npy, crit_alpha.npy, crit_beta.npy, cut_{phi,rho,alpha,beta,nrho}.npy
#   blp_cf_{I,r,s}{n}.npy        closed form at D (I_n, landing radius, emitted p^r sign)
#   blp_cffin_{I,r}{n}.npy       closed form at D_fin
#   blp_rk_{I,r,g,lam,eta}{n}.npy, blp_rk_steps.npy   march at C = 80
#   cut_blp_cf_{I,r}{n}.npy, cut_blp_cffin_{I,r}{n}.npy, cut_blp_rk_{I,r}{n}.npy
#   csweep_C.npy, csweep_idx.npy, csweep_cf_I{n}.npy, csweep_rk_I{n}_C<C>.npy
#   blp_pm_{I,r}{n}.npy, blp_pm_steps.npy, cut_blp_pm_{I,r}{n}.npy   precise march
#   blp_meta.txt, blp_pm_meta.txt   code SHA, configuration, threads, wall times
# and figures/paper/out/cmp_disk/configs.json (read by the AART driver).
# Env: BLP_CMP_CONFIGS (default "A,B"); BLP_CMP_PARTS (default "base,precise": base = the
# closed form, the cuts and the RK4 march; precise = the precise march, which reads the
# cuts of base); BLP_CMP_OUT (output root, default figures/paper/out/cmp_disk).
# Run from the repository root: julia -t 16 --project=. figures/paper/fig_cmp_disk.jl

using BlackLightPlus
using BlackLightPlus.Spacetime: rk4_step, march_control, MarchBudget, camera_ray,
                                inverse_metric, pixel_prep, band_landing, kerr_delta,
                                isco_prograde, kerr_horizon, position, ks_radius
using BlackLightPlus.Matter: johnson_su_profile, intensity
using BlackLightPlus.Plans: precise
using BlackLightPlus.Products: render_crossings
using StaticArrays
using Printf

include(joinpath(@__DIR__, "fig_cmp_disk_common.jl"))

const C_DEFAULT = 80.0
const C_SWEEP = (20.0, 40.0, 80.0, 160.0, 320.0)
const R_EMIT = 1.0e4          # outbound rays beyond this radius stop (emission there < 1e-20 of the peak)

# ------------------------------------------------------------------ RK4 march
# Illinois root of z(rk4_step(y, h)) on (0, H), where z changes sign over the full step.
function crossing_state(st, y::SVector{8,Float64}, H::Float64, yH::SVector{8,Float64})
    a, fa = 0.0, y[4]
    b, fb = H, yH[4]
    yb = yH
    for _ in 1:100
        c = b - fb * (b - a) / (fb - fa)
        yc = rk4_step(st, y, c)
        fc = yc[4]
        if fc * fb < 0
            a, fa = b, fb
        else
            fa /= 2
        end
        b, fb, yb = c, fc, yc
        (fb == 0 || abs(b - a) <= 4 * eps(H) * H) && break
    end
    return yb
end

"""
    crossing_observables(st, yc, disk) -> (r, I, g, λ, η)

The disk at the equatorial crossing state `yc` (`z ≈ 0`): Kerr–Schild (= Boyer–Lindquist)
radius `r`, intensity `I = profile(r) g³`, redshift `g = 1/(p_μ u^μ)` with the
Boyer–Lindquist covector `(p_t, p_r, p_φ)` of the marched `p`: `p_φ = x p_y − y p_x`,
`p_r = r² p^r/Δ` with `p^r = (x ẋ + y ẏ)/r` on the equator, and the photon's `λ = −p_φ/p_t`,
`η = r² p_z²/p_t²` (`p_θ = −r p_z` on the equator).
"""
function crossing_observables(st, yc, disk)
    a = disk.a
    x, yy, z = yc[2], yc[3], yc[4]
    r = ks_radius(st, SVector(yc[1], x, yy, z))
    pt, px, py, pz = yc[5], yc[6], yc[7], yc[8]
    pφ = x * py - yy * px
    ẋ = inverse_metric(st, position(yc)) * SVector(pt, px, py, pz)
    pr_up = (x * ẋ[2] + yy * ẋ[3]) / r
    Δ = kerr_delta(r, a)
    pr = r * r * pr_up / Δ
    λ = -pφ / pt
    η = r * r * pz * pz / (pt * pt)
    r <= kerr_horizon(a) && return r, 0.0, NaN, λ, η
    ut, ur, uφ = disk_velocity(r, disk)
    g = 1 / (pt * ut + pr * ur + pφ * uφ)
    I = johnson_su_profile(r; mu = disk.mu_p, sigma = disk.sigma_p, gamma = disk.gamma_p) *
        g^disk.redshift_power
    return r, I, g, λ, η
end

"""
    march_disk(st, frame, α, β, budget, disk) -> (obs::NTuple{3}, steps)

March the camera ray at `(α, β)` with the fixed-control rule and record the first three
equatorial crossings (orders n = 0, 1, 2); `obs[k] = (r, I, g, λ, η)`, NaN where the ray
stops (capture, escape, or the emission radius) before its k-th crossing.
"""
function march_disk(st, fr, α, β, b::MarchBudget{Float64}, disk)
    nanobs = (NaN, 0.0, NaN, NaN, NaN)
    obs = [nanobs, nanobs, nanobs]
    y, ok = camera_ray(st, fr, α, β)
    ok || return (obs[1], obs[2], obs[3]), -1
    r_prev = Inf
    n = 0; k = 0
    while n < b.nmax
        m = march_control(st, y, r_prev, b)
        m.stop && break
        r_prev = m.R
        yn = rk4_step(st, y, m.h)
        n += 1
        if y[4] != 0 && (yn[4] == 0 || signbit(yn[4]) != signbit(y[4]))
            k += 1
            obs[k] = crossing_observables(st, crossing_state(st, y, m.h, yn), disk)
            k == 3 && break
        end
        y = yn
    end
    return (obs[1], obs[2], obs[3]), n
end

function march_points(st, fr, αs, βs, C, disk; escape)
    b = MarchBudget(C, escape, R_EMIT, 1_000_000)
    np = length(αs)
    out = [Matrix{Float64}(undef, np, 5) for _ in 1:3]
    steps = Vector{Int}(undef, np)
    Threads.@threads :dynamic for i in 1:np
        o, s = march_disk(st, fr, αs[i], βs[i], b, disk)
        for n in 1:3, q in 1:5
            @inbounds out[n][i, q] = o[n][q]
        end
        steps[i] = s
    end
    return out, steps
end

# ------------------------------------------------------------------ closed form on points
function closed_form_points(αs, βs, θo, a, ro, disk)
    np = length(αs)
    I = [zeros(np) for _ in 1:3]; R = [fill(NaN, np) for _ in 1:3]
    Threads.@threads :dynamic for i in 1:np
        p = pixel_prep(αs[i], βs[i], θo, a; ro = ro)
        for n in 0:2
            l = band_landing(p, n)
            if l.valid
                I[n + 1][i] = intensity(disk, l)
                R[n + 1][i] = l.r
            end
        end
    end
    return I, R
end

# ------------------------------------------------------------------ precise march
"""
    run_precise(cfg)

The precise march (`Plans.precise`, `render_crossings`) on the grid nodes and on the radial
cuts of `cfg`, written as `blp_pm_*` and `cut_blp_pm_*` with `blp_pm_meta.txt`. Reads the
cuts written by `run_config`.
"""
function run_precise(cfg)
    dir = cmp_dir(cfg)
    st = KerrSchildCartesian(cfg.a)
    disk = EquatorialDisk(cfg.a)
    nodes = cmp_nodes(cfg)
    camera = CameraConfig(inclination_deg = cfg.inc, fov = cmp_fov(cfg), resolution = cfg.N)
    plan = prepare(precise(RunConfig(camera = camera)); spacetime = st, matter = nothing)
    dxmax = maximum(abs.(plan.pixels.centres .- nodes))
    t_pm = @elapsed pm = render_crossings(plan, disk; orders = (0, 1, 2), backend = :cpu)
    for n in 0:2
        save_npy(joinpath(dir, "blp_pm_I$n.npy"), pm.Ib[n + 1])
        save_npy(joinpath(dir, "blp_pm_r$n.npy"), pm.rs[n + 1])
    end
    save_npy(joinpath(dir, "blp_pm_steps.npy"), pm.steps)
    cα, cβ = load_npy_vector(joinpath(dir, "cut_alpha.npy")), load_npy_vector(joinpath(dir, "cut_beta.npy"))
    t_cut = @elapsed pc = render_crossings(plan, disk; orders = (0, 1, 2), coords = collect(zip(cα, cβ)),
                                           backend = :cpu)
    for n in 0:2
        save_npy(joinpath(dir, "cut_blp_pm_I$n.npy"), pc.Ib[n + 1])
        save_npy(joinpath(dir, "cut_blp_pm_r$n.npy"), pc.rs[n + 1])
    end
    b = plan.geodesic_budget
    open(joinpath(dir, "blp_pm_meta.txt"), "w") do io
        println(io, "git=", cmp_git_sha())
        @printf(io, "config=%s a=%.17g inc=%.17g L=%.17g N=%d camera at infinity\n", cfg.name, cfg.a, cfg.inc,
                cfg.L, cfg.N)
        @printf(io, "fov=%.17g max|centre-node|=%.3e\n", cmp_fov(cfg), dxmax)
        @printf(io, "precise: integrator=%s C=%g capture=%.6g horizon_eps=%g winding=%s nmax=%d escape=%.6g\n",
                nameof(typeof(b.integrator)), b.C, b.capture, b.horizon_eps, b.winding, b.nmax, b.escape)
        @printf(io, "run: threads=%d grid=%.1f s (%.1f steps per ray) cuts=%.1f s on %d points\n",
                Threads.nthreads(), t_pm, sum(pm.steps) / length(pm.steps), t_cut, length(cα))
    end
    @printf("[%s] precise march %d^2: %.1f s, %.1f steps per ray; cuts %.1f s\n", cfg.name, cfg.N, t_pm,
            sum(pm.steps) / length(pm.steps), t_cut)
end

# ------------------------------------------------------------------ one configuration
function run_config(cfg)
    dir = cmp_dir(cfg)
    a = cfg.a; θo = deg2rad(cfg.inc)
    st = KerrSchildCartesian(a)
    disk = EquatorialDisk(a)
    nodes = cmp_nodes(cfg)
    save_npy(joinpath(dir, "nodes.npy"), nodes)

    αc, βc = critical_curve(a, θo; n = 4000)
    save_npy(joinpath(dir, "crit_alpha.npy"), αc)
    save_npy(joinpath(dir, "crit_beta.npy"), βc)
    phi, rho, cα, cβ, nrho = cmp_cuts(cfg, φ -> loop_radius(αc, βc, φ))
    for (nm, v) in (("phi", phi), ("rho", rho), ("alpha", cα), ("beta", cβ), ("nrho", Float64.(nrho)))
        save_npy(joinpath(dir, "cut_$nm.npy"), v)
    end

    # closed form on the grid at D and at D_fin
    camera = CameraConfig(inclination_deg = cfg.inc, distance = cfg.D_rk, fov = cmp_fov(cfg),
                          resolution = cfg.N)
    plan = prepare(RunConfig(camera = camera); spacetime = st, matter = nothing)
    centres = plan.pixels.centres
    dxmax = maximum(abs.(centres .- nodes))
    t_cf = @elapsed img = render_analytic(plan, disk; bands = (0, 1, 2), ro = cfg.D, backend = :cpu)
    t_cfi = @elapsed imgi = render_analytic(plan, disk; bands = (0, 1, 2), ro = cfg.D_fin, backend = :cpu)
    for n in 0:2
        save_npy(joinpath(dir, "blp_cf_I$n.npy"), img.Ib[n + 1])
        save_npy(joinpath(dir, "blp_cf_r$n.npy"), img.rs[n + 1])
        save_npy(joinpath(dir, "blp_cf_s$n.npy"), img.redshift_sign[n + 1])
        save_npy(joinpath(dir, "blp_cffin_I$n.npy"), imgi.Ib[n + 1])
        save_npy(joinpath(dir, "blp_cffin_r$n.npy"), imgi.rs[n + 1])
    end
    @printf("[%s] closed form %d^2: %.1f s (D), %.1f s (D_fin); max |centre - node| = %.2e\n",
            cfg.name, cfg.N, t_cf, t_cfi, dxmax)

    # closed form on the cuts
    t_cut = @elapsed begin
        cI, cR = closed_form_points(cα, cβ, θo, a, cfg.D, disk)
        cIi, cRi = closed_form_points(cα, cβ, θo, a, cfg.D_fin, disk)
    end
    for n in 0:2
        save_npy(joinpath(dir, "cut_blp_cf_I$n.npy"), cI[n + 1])
        save_npy(joinpath(dir, "cut_blp_cf_r$n.npy"), cR[n + 1])
        save_npy(joinpath(dir, "cut_blp_cffin_I$n.npy"), cIi[n + 1])
        save_npy(joinpath(dir, "cut_blp_cffin_r$n.npy"), cRi[n + 1])
    end

    # RK4 march on the grid: pixel k of the camera (β fastest) at (centres[iα], centres[iβ])
    fr = plan.frame
    N = cfg.N
    gα = vec([centres[j] for i in 1:N, j in 1:N])          # column-major [iβ, iα]
    gβ = vec([centres[i] for i in 1:N, j in 1:N])
    t_rk = @elapsed out, steps = march_points(st, fr, gα, gβ, C_DEFAULT, disk; escape = plan.escape)
    for n in 0:2, (q, nm) in enumerate(("r", "I", "g", "lam", "eta"))
        save_npy(joinpath(dir, "blp_rk_$nm$n.npy"), reshape(out[n + 1][:, q], N, N))
    end
    save_npy(joinpath(dir, "blp_rk_steps.npy"), reshape(Float64.(steps), N, N))
    @printf("[%s] RK4 march %d^2 at C = %g: %.1f s, %.0f steps per ray on average\n",
            cfg.name, N, C_DEFAULT, t_rk, sum(steps) / length(steps))

    # RK4 march on the cuts
    t_rkc = @elapsed cout, _ = march_points(st, fr, cα, cβ, C_DEFAULT, disk; escape = plan.escape)
    for n in 0:2
        save_npy(joinpath(dir, "cut_blp_rk_I$n.npy"), cout[n + 1][:, 2])
        save_npy(joinpath(dir, "cut_blp_rk_r$n.npy"), cout[n + 1][:, 1])
    end

    # step-size sweep on every fifteenth cut (all radii of the angles k = 1, 16, 31, …)
    offs = cumsum(vcat(0, Int.(nrho)))
    idx = reduce(vcat, [collect(offs[k] + 1:offs[k + 1]) for k in 1:15:length(phi)])
    save_npy(joinpath(dir, "csweep_idx.npy"), Float64.(idx))
    save_npy(joinpath(dir, "csweep_C.npy"), collect(C_SWEEP))
    for n in 0:2
        save_npy(joinpath(dir, "csweep_cf_I$n.npy"), cI[n + 1][idx])
    end
    t_sw = @elapsed for C in C_SWEEP
        so, _ = march_points(st, fr, cα[idx], cβ[idx], C, disk; escape = plan.escape)
        for n in 0:2
            save_npy(joinpath(dir, "csweep_rk_I$(n)_C$(Int(C)).npy"), so[n + 1][:, 2])
            save_npy(joinpath(dir, "csweep_rk_r$(n)_C$(Int(C)).npy"), so[n + 1][:, 1])
        end
    end

    open(joinpath(dir, "blp_meta.txt"), "w") do io
        println(io, "git=", cmp_git_sha())
        @printf(io, "config=%s a=%.17g inc=%.17g L=%.17g N=%d D=%.17g D_fin=%.17g D_rk=%.17g\n",
                cfg.name, cfg.a, cfg.inc, cfg.L, cfg.N, cfg.D, cfg.D_fin, cfg.D_rk)
        @printf(io, "fov=%.17g max|centre-node|=%.3e\n", cmp_fov(cfg), dxmax)
        @printf(io, "disk: EquatorialDisk(%.17g) sub_kep=%g betar=%g betaphi=%g gamma_p=%g mu_p=%.17g sigma_p=%g power=%g\n",
                a, disk.sub_kep, disk.betar, disk.betaphi, disk.gamma_p, disk.mu_p, disk.sigma_p, disk.redshift_power)
        @printf(io, "march: C=%g escape=%.6g R_em=%.6g; sweep C=%s on %d cut points\n",
                C_DEFAULT, plan.escape, R_EMIT, string(C_SWEEP), length(idx))
        @printf(io, "cuts: %d angles, %d points\n", length(phi), length(cα))
        @printf(io, "run: threads=%d cf=%.1f s cf_fin=%.1f s cuts_cf=%.1f s rk_grid=%.1f s rk_cuts=%.1f s sweep=%.1f s\n",
                Threads.nthreads(), t_cf, t_cfi, t_cut, t_rk, t_rkc, t_sw)
    end
    @printf("[%s] cuts: %d points; sweep %.1f s\n", cfg.name, length(cα), t_sw)
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_configs_json()
    parts = strip.(split(get(ENV, "BLP_CMP_PARTS", "base,precise"), ","))
    for name in split(get(ENV, "BLP_CMP_CONFIGS", "A,B"), ",")
        cfg = cmp_config(strip(name))
        "base" in parts && run_config(cfg)
        "precise" in parts && run_precise(cfg)
    end
end
