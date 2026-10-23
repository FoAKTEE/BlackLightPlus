# fig_truecolor.jl — multi-frequency images for the true-colour figure (paper figure 15):
# the fiducial deck viewed face-on (θ = 0), at θ = 45° and edge-on (θ = 90°), Stokes I at
# 12 frequencies evenly spaced in wavelength from 152 to 324 GHz, plus the 230 GHz image.
# Writes out/blacklight/truecolor/i<θ>_f<k>_I.npy, i<θ>_230_I.npy and record.json; the
# plotter boosts the spectra into the visible band and integrates the colour-matching
# functions.
# Run from the repository root:   julia -t 32 --project=BLP BLP/figures/blacklight/fig_truecolor.jl
include(joinpath(@__DIR__, "common.jl"))

function main()
    out = outdir("truecolor")
    snap = AthenaKSnapshot(dump_path())
    kerr = KerrSchildCartesian(BL_SPIN)
    λs = range(CL / 324e9, CL / 152e9; length = 12)
    νs = CL ./ λs
    rec = Dict{String,Any}("frequencies_hz" => collect(νs))
    for inc in (0.0, 45.0, 90.0)
        t0 = time()
        for (k, ν) in enumerate(νs)
            img = render(prepare(fiducial_config(; inc = inc, nu = ν); spacetime = kerr, matter = snap, mode = StokesI()); backend = :cpu)
            write_npy(joinpath(out, @sprintf("i%02d_f%02d_I.npy", round(Int, inc), k)), img.I)
        end
        img = render(prepare(fiducial_config(; inc = inc); spacetime = kerr, matter = snap, mode = StokesI()); backend = :cpu)
        write_npy(joinpath(out, @sprintf("i%02d_230_I.npy", round(Int, inc))), img.I)
        rec["inc$(round(Int, inc))"] = Dict("seconds" => time() - t0, "flux_230_jy" => flux_jy(img.I, BL_FOV, BL_RES))
        logf("inclination %.0f: 13 images in %.0f s, 230 GHz flux %.3f Jy\n", inc, time() - t0, rec["inc$(round(Int, inc))"]["flux_230_jy"])
    end
    write_json(joinpath(out, "record.json"), record("truecolor"; result = rec))
end
main()
