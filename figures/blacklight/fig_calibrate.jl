# fig_calibrate.jl — the density mass scale of the fiducial deck: the scale at which the
# fiducial image (paper camera, 128^2 here) carries 2.4 Jy at 230 GHz from 8.178 kpc,
# found on a mass-scale ladder of `variants` (one trace, 24 images) by log-log
# interpolation and confirmed by one render at 256^2. Writes out/blacklight/mass_scale.txt
# and calibrate/record.json.
# Run from the repository root:   julia -t 32 --project=. figures/blacklight/fig_calibrate.jl
include(joinpath(@__DIR__, "common.jl"))

function main()
    out = outdir("calibrate")
    snap = AthenaKSnapshot(dump_path())
    L3 = BL_L^3
    ladder = [10.0^e * L3 for e in range(-23.0, -17.0; length = 25)]
    cfg = fiducial_config(; res = 128, mass = ladder[1])
    plan = prepare(cfg; spacetime = KerrSchildCartesian(BL_SPIN), matter = snap, mode = StokesI())
    t0 = time()
    v = variants(plan, ladder; backend = :cpu)
    fl = [flux_jy(v.images[k], BL_FOV, 128) for k in 1:length(ladder)]
    logf("ladder in %.1f s\n", time() - t0)
    for (m, f) in zip(ladder, fl)
        logf("  mass scale %.3e g (%.1e L^3)  flux %.4f Jy\n", m, m / L3, f)
    end
    # log-log interpolation of the first crossing of 2.4 Jy
    k = findfirst(i -> fl[i] >= BL_FLUX_JY, 1:length(fl))
    (k === nothing || k == 1) && error("the ladder does not bracket $(BL_FLUX_JY) Jy")
    lm = log(ladder[k-1]) + (log(BL_FLUX_JY) - log(fl[k-1])) / (log(fl[k]) - log(fl[k-1])) * (log(ladder[k]) - log(ladder[k-1]))
    ms = exp(lm)
    cfg2 = fiducial_config(; mass = ms)
    img = render(prepare(cfg2; spacetime = KerrSchildCartesian(BL_SPIN), matter = snap, mode = StokesI()); backend = :cpu)
    f2 = flux_jy(img.I, BL_FOV, BL_RES)
    ms2 = ms * (BL_FLUX_JY / f2)^(1 / (log(fl[k]) - log(fl[k-1])) * (log(ladder[k]) - log(ladder[k-1])) * 1.0)
    logf("interpolated mass scale %.4e g (%.3e L^3): flux at 256^2 %.4f Jy; corrected %.4e\n", ms, ms / L3, f2, ms2)
    img2 = render(prepare(fiducial_config(; mass = ms2); spacetime = KerrSchildCartesian(BL_SPIN), matter = snap, mode = StokesI()); backend = :cpu)
    f3 = flux_jy(img2.I, BL_FOV, BL_RES)
    logf("flux at the corrected scale %.4f Jy\n", f3)
    open(joinpath(BL_OUT, "mass_scale.txt"), "w") do io
        println(io, ms2)
    end
    write_json(joinpath(out, "record.json"),
               record("calibrate"; ladder = ladder, flux_jy = fl, mass_scale = ms2,
                      mass_scale_L3 = ms2 / L3, flux_256 = f3, target_jy = BL_FLUX_JY))
end
main()
