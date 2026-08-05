# fig_examples.jl — minimal demonstration renders for BlackLightPlus.
#   (a) Exact-Kerr analytic tracer: AART-style equatorial disk (EquatorialDisk)
#       at a = 0.94, i = 17 deg, photon-ring bands n = 0,1,2 -> total intensity.
#   (b) Polarized coherency-tensor march on an analytic plasma source: a
#       PolishDoughnut torus bridged into the GRMHD primitive seam
#       (source_plasma_at -> normal-frame uu/bb with unit d/e/b scales) -> IQUV.
# Writes NumPy .npy arrays to figures/out/ (plot with plot_examples.py).
# Resolution override: BLP_FIG_RES (default 256; polarized panel capped at 128).
# Run: julia --project=. -t auto figures/fig_examples.jl

using BlackLightPlus
using BlackLightPlus: polarized_march
using BlackLightPlus.Constants: MP, ME, CL
using StaticArrays
using Printf

const RES = parse(Int, get(ENV, "BLP_FIG_RES", "256"))
const OUT = joinpath(@__DIR__, "out")
mkpath(OUT)

const A = 0.94                                   # black-hole spin for both panels

# ------------------------------------------------------- (a) analytic Kerr disk
cfg = RunConfig(bh_a=A, tracer=:analytic,
                camera=CameraConfig(inclination_deg=17.0, distance=1.0e4,
                                    fov=20.0, resolution=RES))
ta = @elapsed img = render_analytic(cfg; bands=(0, 1, 2), source=EquatorialDisk(A))
write_npy(joinpath(OUT, "analytic_disk_I.npy"), img.I)
@printf("analytic disk : %4d^2  bands (0,1,2)  %6.1f s  sum(I) = %.4e\n",
        RES, ta, sum(img.I))

# --------------------------------------------------- (b) polarized torus (IQUV)
# source_plasma_at returns cgs (n_e, theta_e, B) plus exact 4-vectors; with unit
# d/e/b scales and R_low = R_high = 1 the plasma decode inverts in closed form,
# so the analytic torus drives the same polarized march as a GRMHD snapshot.
function source_sampler(src, st)
    vac = (rho=0.0, pgas=0.0, uu=SVector(0.0, 0.0, 0.0),
           bb=SVector(0.0, 0.0, 0.0), inside=false)
    return x -> begin
        pl = source_plasma_at(src, st, x)
        (pl.inside && pl.B > 0.0) || return vac
        u = pl.ucon
        g0 = pl.gcon
        # normal-frame velocity uu^i = u^i - (g^{0i}/g^{00}) u^0
        uu = SVector(u[2] - g0[1, 2] / g0[1, 1] * u[1],
                     u[3] - g0[1, 3] / g0[1, 1] * u[1],
                     u[4] - g0[1, 4] / g0[1, 1] * u[1])
        # lab-frame 3-field B^i = b^i u^0 - b^0 u^i from b^mu = |B| bcon
        b = pl.B * pl.bcon
        bb = SVector(b[2] * u[1] - b[1] * u[2],
                     b[3] * u[1] - b[1] * u[3],
                     b[4] * u[1] - b[1] * u[4])
        return (rho=pl.n_e * MP, pgas=2.0 * pl.n_e * pl.theta_e * ME * CL * CL,
                uu=uu, bb=bb, inside=true)
    end
end

const PRES = min(128, RES)
torus = PolishDoughnut(A; r_center=9.0, n_e_center=5.0e5, theta_e_center=40.0,
                       beta=8.0)
st = KerrSchildCartesian(A)
inc_deg = 60.0
dist = 1000.0
fovp = 2.2 * torus.r_out
nu_obs = 230.0e9
L_unit = 9.6e14                                  # GM/c^2 of an M87*-mass hole [cm]

tb = @elapsed begin
    cam = PlaneParallelCamera(inc_deg, dist, fovp, PRES)
    rays = camera_rays(st, cam)
    states, dls = trace_rays_sampled(st, rays; C=80.0, nmax=2500, escape=1.1 * dist)
    scc = SimCoeffConfig(bh_a=A, d_unit=1.0, e_unit=1.0, b_unit=1.0,
                         plasma_rat_low=1.0, plasma_rat_high=1.0)
    sampler = source_sampler(torus, st)
    i_rad = deg2rad(inc_deg)
    up = SVector(0.0, -cos(i_rad), 0.0, sin(i_rad))   # camera vertical (image beta)
    S = Matrix{Float64}(undef, 4, length(rays))
    Threads.@threads for i in 1:length(rays)
        S[:, i] .= polarized_march(st, scc, sampler, states[i], dls[i],
                                   nu_obs, L_unit; cam_up=up)
    end
end
for (k, name) in enumerate(("I", "Q", "U", "V"))
    write_npy(joinpath(OUT, "torus_$(name).npy"), reshape(S[k, :], PRES, PRES))
end
@printf("polarized torus: %4d^2  r_out = %.1f M  %6.1f s  sum(I) = %.4e  max|P|/max I = %.3f\n",
        PRES, torus.r_out, tb,
        sum(S[1, :]), maximum(hypot.(S[2, :], S[3, :])) / maximum(S[1, :]))
