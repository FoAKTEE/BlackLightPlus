# fig_formula.jl — the five parameterized models of Gold et al. 2020 (paper figure 3 and
# Mahakala figure 7): BLP traces the camera rays (fixed-control march, step
# (r - r_h)/C, C = 100), the script evaluates the model along each stored path and
# integrates the unpolarized transfer source → camera with BLP's exact segment operator.
# Model (Gold et al. 2020 eqs. 5-12, as Blacklight formula_coefficients.cpp): in
# Boyer-Lindquist (r, θ) of the Kerr-Schild radius, n/n0 = exp(-½(r²/r0² + h² cos²θ)),
# l = l0 (r sinθ)^(1+q)/(1 + r sinθ), u_t = -u_n, u_φ = u_n l with u_n normalising u,
# u^μ raised with the BL inverse and carried to Kerr-Schild Cartesian components;
# j_ν = C n0 n (ν/ν_p)^(-α), α_ν = A C n0 n (ν/ν_p)^(-α-β) with ν the fluid-frame
# frequency -k·u ν_obs. Parameters: r0 = 10, q = 1/2, ν_p = 230 GHz, C n0 = 3e-18
# erg s⁻¹ cm⁻³ sr⁻¹ Hz⁻¹, β = 5/2, GM/c² = 6e11 cm; per test (A, α, h, l0, a) from
# Gold et al. Table 1. Camera: plane-parallel at r = 1000 M, inclination 60°, width
# 30 M, 128², 230 GHz; the source distance is taken as 1 M = 5 μas (the axes of the
# published figures), D = 2.4752e22 cm. Writes out/blacklight/formula/test<k>_I.npy and
# record.json (fluxes in Jy against the exact values of Gold et al. Table 2).
# Run from the repository root:   julia -t 16 --project=BLP BLP/figures/blacklight/fig_formula.jl
include(joinpath(@__DIR__, "common.jl"))
using BLP.Transfer: segment

const TESTS = [(A = 0.0, alpha = -3.0, h = 0.0, l0 = 0.0, a = 0.9),
               (A = 0.0, alpha = -2.0, h = 0.0, l0 = 1.0, a = 0.0),
               (A = 0.0, alpha = 0.0, h = 10 / 3, l0 = 1.0, a = 0.9),
               (A = 1.0e5, alpha = 0.0, h = 10 / 3, l0 = 1.0, a = 0.9),
               (A = 1.0e6, alpha = 0.0, h = 100 / 3, l0 = 1.0, a = 0.9)]
const EXACT = [1.6465, 1.4360, 0.4418, 0.2710, 0.0255]
const R0 = 10.0; const QQ = 0.5; const NUP = 2.3e11; const CN0 = 3.0e-18; const BETA = 2.5
const M_CM = 6.0e11
const D_CM = M_CM / (5.0e-6 / 206264.806)
const F_FOV = 30.0; const F_RES = 128; const F_RCAM = 1000.0; const F_INC = 60.0

# invariant (j, α) of the model at the ray state y for test t
function model_coefficients(t, y, ν_obs)
    a = t.a
    x, yy, z = y[2], y[3], y[4]
    k = -SVector(y[5], y[6], y[7], y[8])                           # wave covector, k_t = -1
    rr2 = x * x + yy * yy + z * z
    r2 = (rr2 - a * a + sqrt((rr2 - a * a)^2 + 4 * a * a * z * z)) / 2
    r = sqrt(r2)
    rr = sqrt(max(r * r - z * z, 0.0))
    cth = z / r; sth = sqrt(max(1 - cth * cth, 0.0))
    ph = atan(yy, x) - atan(a / r)
    sph = sin(ph); cph = cos(ph)
    delta = r * r - 2r + a * a
    sigma = r * r + a * a * cth * cth
    gtt = -(1 + 2r * (r * r + a * a) / (delta * sigma))
    gtph = -2a * r / (delta * sigma)
    gphph = (sigma - 2r) / (delta * sigma * sth * sth)
    ll = t.l0 / (1 + rr) * rr^(1 + QQ)
    un = 1 / sqrt(-gtt + 2gtph * ll - gphph * ll * ll)
    u_t = -un; u_ph = un * ll
    ut = gtt * u_t + gtph * u_ph
    uph = gtph * u_t + gphph * u_ph                                 # u^r = u^θ = 0
    u0 = ut
    u1 = sth * (-r * sph - a * cph) * uph
    u2 = sth * (r * cph - a * sph) * uph
    u3 = 0.0
    n = exp(-0.5 * (r * r / (R0 * R0) + t.h * t.h * cth * cth))
    nu = -(u0 * k[1] + u1 * k[2] + u2 * k[3] + u3 * k[4]) * ν_obs
    j_nu = CN0 * n * (nu / NUP)^(-t.alpha)
    a_nu = t.A * CN0 * n * (nu / NUP)^(-BETA - t.alpha)
    return j_nu / (nu * nu), a_nu * nu
end

function main()
    out = outdir("formula")
    m_bh = M_CM * CL^2 / GNEWT
    rec = Dict{String,Any}()
    for (k, t) in enumerate(TESTS)
        cfg = RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0, frequency_hz = BL_NU,
                        camera = CameraConfig(inclination_deg = F_INC, distance = F_RCAM, fov = F_FOV, resolution = F_RES),
                        max_steps = 40000, C = 100.0, escape_factor = 1.05)
        plan = prepare(cfg; spacetime = KerrSchildCartesian(t.a), matter = nothing, mode = StokesI())
        t0 = time()
        traj = trace(plan; backend = :cpu)
        ν = BL_NU
        I = zeros(F_RES, F_RES)
        nst = zeros(Int, F_RES, F_RES)
        Threads.@threads for px in 1:F_RES^2
            ys = traj.states[px]; dls = traj.dls[px]
            Iinv = 0.0
            for n in 1:length(ys)-1
                jj, aa = model_coefficients(t, ys[n], ν)
                ds = dls[n] * M_CM / ν                              # cgs affine length
                A_, B_ = segment(StokesI(), jj, aa, ds)
                Iinv = A_ * Iinv + B_
            end
            ib, ia = (px - 1) % F_RES + 1, (px - 1) ÷ F_RES + 1  # camera order, β fastest
            I[ib, ia] = Iinv * ν^3
            nst[ib, ia] = length(ys)
        end
        pix = (F_FOV / F_RES * M_CM / D_CM)^2
        flux = sum(I) * pix * 1e23
        write_npy(joinpath(out, "test$(k)_I.npy"), I)
        rec["test$k"] = Dict("params" => Dict(string(kk) => v for (kk, v) in pairs(t)), "flux_jy" => flux,
                             "exact_jy" => EXACT[k], "ratio" => flux / EXACT[k], "seconds" => time() - t0,
                             "mean_steps" => sum(nst) / length(nst))
        logf("test %d: flux %.4f Jy (exact %.4f, ratio %.4f), %.0f s, %.0f steps per ray\n", k, flux, EXACT[k],
             flux / EXACT[k], time() - t0, sum(nst) / length(nst))
    end
    write_json(joinpath(out, "record.json"), record("formula"; tests = rec, r0 = R0, q = QQ, nu_p = NUP, cn0 = CN0,
                        beta = BETA, m_cm = M_CM, d_cm = D_CM, fov = F_FOV, res = F_RES, rcam = F_RCAM, inc = F_INC))
end
main()
