# fig_nonthermal.jl — electron distributions (paper figure 10): the fiducial thermal image
# and the images of a power law (p = 3, γ = 4 … 1000), a kappa distribution (κ = 4,
# w = 1/2) and the equal three-way mix, each with its density mass scale set on a
# `variants` ladder so that its total flux matches the fiducial 2.4 Jy, as the paper
# chooses its nonthermal parameters to match the thermal flux. Writes
# out/blacklight/nonthermal/<tag>_I.npy and record.json (the matched mass scales).
# Run from the repository root:   julia -t 32 --project=BLP BLP/figures/blacklight/fig_nonthermal.jl
include(joinpath(@__DIR__, "common.jl"))

function matched_scale(cfg, snap, kerr)
    ladder = [cfg.mass_scale_cgs * 10.0^e for e in range(-1.0, 2.5; length = 29)]
    v = variants(prepare(cfg; spacetime = kerr, matter = snap, mode = StokesI()), ladder; backend = :cpu)
    fl = [flux_jy(v.images[k], BL_FOV, cfg.camera.resolution) for k in 1:length(ladder)]
    k = findfirst(i -> fl[i] >= BL_FLUX_JY, 1:length(fl))
    (k === nothing || k == 1) && error("ladder does not bracket the flux: $(extrema(fl))")
    lm = log(ladder[k-1]) + (log(BL_FLUX_JY) - log(fl[k-1])) / (log(fl[k]) - log(fl[k-1])) * (log(ladder[k]) - log(ladder[k-1]))
    return exp(lm)
end

function main()
    out = outdir("nonthermal")
    snap = AthenaKSnapshot(dump_path())
    kerr = KerrSchildCartesian(BL_SPIN)
    decks = (("thermal", PlasmaConfig()),
             ("powerlaw", PlasmaConfig(thermal_frac = 0.0, power_frac = 1.0, power_p = 3.0, gamma_min = 4.0, gamma_max = 1000.0)),
             ("kappa", PlasmaConfig(thermal_frac = 0.0, kappa_frac = 1.0, kappa = 4.0, w = 0.5)),
             ("mixed", PlasmaConfig(thermal_frac = 1 / 3, power_frac = 1 / 3, kappa_frac = 1 / 3, power_p = 3.0,
                                    gamma_min = 4.0, gamma_max = 1000.0, kappa = 4.0, w = 0.5)))
    rec = Dict{String,Any}()
    for (tag, pl) in decks
        t0 = time()
        ms = tag == "thermal" ? mass_scale() : matched_scale(fiducial_config(; res = 128, plasma = pl), snap, kerr)
        img = render(prepare(fiducial_config(; plasma = pl, mass = ms); spacetime = kerr, matter = snap, mode = StokesI()); backend = :cpu)
        write_npy(joinpath(out, "$(tag)_I.npy"), img.I)
        rec[tag] = Dict("mass_scale" => ms, "mass_scale_ratio" => ms / mass_scale(), "flux_jy" => flux_jy(img.I, BL_FOV, BL_RES),
                        "seconds" => time() - t0)
        logf("%s: mass scale %.3e (%.2f x fiducial), %.3f Jy, %.0f s\n", tag, ms, ms / mass_scale(), rec[tag]["flux_jy"], rec[tag]["seconds"])
    end
    write_json(joinpath(out, "record.json"), record("nonthermal"; runs = rec, power_p = 3.0, gamma_min = 4.0, gamma_max = 1000.0, kappa = 4.0, w = 0.5))
end
main()
