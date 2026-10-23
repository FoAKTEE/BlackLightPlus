# fig_amr.jl — unrefined and adaptively refined images (paper figure 9): the fiducial deck
# at a root resolution of 32² pixels in 8² blocks, refined up to three levels where the
# absolute pixel-to-pixel intensity gradient exceeds 5e-5 cgs on five or more pixels
# (AbsoluteGradient(5e-5; count = 5)), Stokes I and V from `render_amr`; the optical
# depth τ = ∫ α_I dλ from the traced rays sampled along their paths at the root
# resolution and at the effective 256² (the plotter composes the refined τ from the level
# map). Writes out/blacklight/amr/{root_I,root_V,root_tau,amr_I,amr_V,fine_tau,levels}.npy
# and record.json.
# Run from the repository root:   julia -t 32 --project=BLP BLP/figures/blacklight/fig_amr.jl
include(joinpath(@__DIR__, "common.jl"))
using BLP.Analysis: AbsoluteGradient

function tau_map(plan)
    traj = trace(plan; backend = :cpu)
    n = plan.camera.npix
    τ = zeros(n, n)
    Threads.@threads for px in 1:n^2
        s = sampled_path(plan, traj.states[px], traj.dls[px])
        tot = 0.0
        for k in 1:length(s.alpha)
            s.valid[k] && (tot += s.alpha[k] * s.dlam[k] * plan.params.units.L_unit / plan.params.nu_obs)
        end
        τ[(px - 1) % n + 1, (px - 1) ÷ n + 1] = tot
    end
    return τ
end

function main()
    out = outdir("amr")
    snap = AthenaKSnapshot(dump_path())
    kerr = KerrSchildCartesian(BL_SPIN)
    cfg = fiducial_config(; res = 32)
    plan = prepare(cfg; spacetime = kerr, matter = snap, mode = IQUV())
    t0 = time()
    root = render(plan; backend = :cpu)
    t_root = time() - t0
    t0 = time()
    amr = render_amr(plan; block = 8, max_level = 3, refine = AbsoluteGradient(5e-5; count = 5), backend = :cpu)
    t_amr = time() - t0
    write_npy(joinpath(out, "root_I.npy"), root.I); write_npy(joinpath(out, "root_V.npy"), root.V)
    write_npy(joinpath(out, "amr_I.npy"), amr.image.I); write_npy(joinpath(out, "amr_V.npy"), amr.image.V)
    write_npy(joinpath(out, "levels.npy"), Float64.(amr.levels))
    logf("root 32^2 in %.1f s; amr %d rays in %.1f s; levels %s\n", t_root, amr.nrays, t_amr, string([count(==(l), amr.levels) for l in 0:3]))
    t0 = time()
    τ_root = tau_map(prepare(cfg; spacetime = kerr, matter = snap, mode = StokesI()))
    τ_fine = tau_map(prepare(fiducial_config(; res = 256); spacetime = kerr, matter = snap, mode = StokesI()))
    write_npy(joinpath(out, "root_tau.npy"), τ_root); write_npy(joinpath(out, "fine_tau.npy"), τ_fine)
    logf("tau maps in %.0f s\n", time() - t0)
    write_json(joinpath(out, "record.json"), record("amr"; root_seconds = t_root, amr_seconds = t_amr, amr_rays = amr.nrays,
                        levels_hist = [count(==(l), amr.levels) for l in 0:3], abs_grad_cut = 5e-5, count = 5,
                        root_flux = flux_jy(root.I, BL_FOV, 32), amr_flux = flux_jy(amr.image.I, BL_FOV, 256)))
end
main()
