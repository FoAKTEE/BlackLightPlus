# r02_main_check.jl — RAPTOR's full image pipeline on the analytic scenes: the fork's
# RAPTOR executable (camera blocks, OpenMP block loop, compute_spec, HDF5 writer) built in
# a run directory from run/ and model/blp_analytic, against the single-pixel library at
# the same image points (R02 support check).
#   Gold T4 (unpolarized build) and the orientation ring (polarized build), i = 60,
#   100^2, FOV 30, C = 80.
# Usage: julia --project=. -t N r02_main_check.jl <fork tree> <bindir> <out.json>
# Exit 0 iff every HDF5 pixel equals the library value to <= 1e-12 relative (of the image
# maximum) and the spectrum-file flux equals the HDF5 flux (sum I x pixel solid angle)
# to <= 1e-12.

using HDF5, Printf
include(joinpath(@__DIR__, "jsonw.jl"))

const TREE = abspath(ARGS[1])
const BIN = abspath(ARGS[2])
const OUT = ARGS[3]
const NPIX = 100
const FOV = 30.0
const NU = 230.0e9
const CL = 2.99792458e10

function build(dir, pol)
    mkpath(dir)
    for f in readdir(joinpath(TREE, "run")); cp(joinpath(TREE, "run", f), joinpath(dir, f); force = true); end
    for f in readdir(joinpath(TREE, "model", "blp_analytic"))
        cp(joinpath(TREE, "model", "blp_analytic", f), joinpath(dir, f); force = true)
    end
    mk = read(joinpath(dir, "makefile"), String)
    write(joinpath(dir, "makefile"), replace(mk, "CFLAGS = -fopenmp" => "CFLAGS = -DPOL=$(pol) -fopenmp"))
    run(pipeline(Cmd(`make all`; dir = dir, env = merge(ENV, Dict("RAPTOR" => TREE, "PWD" => dir))); stdout = devnull, stderr = devnull))
end

function case(name, pol, mbh_msun, m_unit, dist_kpc, spin, scene)
    dir = joinpath(mktempdir(), name)
    build(dir, pol)
    write(joinpath(dir, "model.in"), """
MBH (Msun) $(repr(mbh_msun))
DISTANCE (kpc) $(repr(dist_kpc))
M_UNIT (g) $(repr(m_unit))
R_LOW (-) 1
R_HIGH (-) 1
INCLINATION (deg) 60.0
IMG_WIDTH (pixels) $NPIX
IMG_HEIGHT (pixels) $NPIX
CAM_SIZE_X (Rg) $FOV
CAM_SIZE_Y (Rg) $FOV
FREQS_PER_DEC (-) 1
FREQ_MIN (Hz) $(repr(NU))
STEPSIZE (-) 0.01
MAX_LEVEL (-) 1
""")
    write(joinpath(dir, "bbh.in"), "nholes 1\nmass 1.0\nspin $(repr(spin))\ncenter1 0 0 0\nC 80.0\nnmax 4000\nescape 1100.0\ncam_dist 1000.0\n")
    write(joinpath(dir, "scene.in"), join(scene, "\n") * "\n")
    run(pipeline(Cmd(`./RAPTOR model.in none 0`; dir = dir,
                     env = merge(ENV, Dict("OMP_NUM_THREADS" => "32", "OMP_STACKSIZE" => "20m",
                                           "RAPTOR_BBH" => "bbh.in", "RAPTOR_SCENE" => "scene.in")));
                 stdout = devnull, stderr = devnull))
    # library at the HDF5 pixel coordinates
    lib = Libc.Libdl.dlopen(joinpath(BIN, pol == 1 ? "libraptor_rt_pol.so" : "libraptor_rt_unpol.so"))
    withenv("RAPTOR_BBH" => joinpath(dir, "bbh.in"), "RAPTOR_SCENE" => joinpath(dir, "scene.in")) do
        redirect_stdout(devnull) do
            ccall(Libc.Libdl.dlsym(lib, :raptor_rt_setup), Cvoid, (Cstring,), joinpath(dir, "model.in"))
        end
    end
    f = h5open(joinpath(dir, "output", "img_data_0.h5"))
    a = vec(read(f["alpha"])); b = vec(read(f["beta"]))
    keys_ = [k for k in keys(f) if k[1] in "IQUV"]
    img = Dict(k[1] => vec(read(f[k])) for k in keys_)
    close(f)
    px = FOV / NPIX
    G = 6.6743e-8; MSUN = 1.98847e33; KPC = 3.0856775814913673e21
    pixsr = (px * G * mbh_msun * MSUN / CL^2 / (dist_kpc * KPC))^2
    fpix = Libc.Libdl.dlsym(lib, :raptor_rt_pixel)
    L = zeros(4, length(a))
    Threads.@threads for i in eachindex(a)
        s = zeros(4)
        ccall(fpix, Cint, (Cdouble, Cdouble, Ptr{Float64}), a[i], b[i], s)
        L[:, i] .= s .* 1e23                     # HDF5 stores Jy sr^-1 (io.c: dA = 1)
    end
    res = Dict{String,Any}()
    ok = true
    for (q, c) in enumerate("IQUV")
        haskey(img, c) || continue
        d = maximum(abs.(img[c] .- L[q, :])) / maximum(abs.(L[1, :]))
        res["max_pixel_diff_$(c)_over_Imax"] = d
        ok &= d <= 1e-12
    end
    spec = parse(Float64, split(readlines(joinpath(dir, "output", "spectrum_0_60.00.dat"))[1])[2])
    res["spectrum_flux_Jy"] = spec
    res["hdf5_flux_Jy"] = sum(img['I']) * pixsr
    res["spectrum_vs_hdf5"] = spec / res["hdf5_flux_Jy"] - 1
    ok &= abs(res["spectrum_vs_hdf5"]) <= 1e-12
    res["pass"] = ok
    @printf("%-5s  max|HDF5 - library|/Imax  I %.2e  spectrum %.6e Jy vs HDF5 sum %.6e  %s\n", name,
            res["max_pixel_diff_I_over_Imax"], spec, res["hdf5_flux_Jy"], ok ? "PASS" : "FAIL")
    return res
end

GL = 6.0e11
r = Dict{String,Any}()
r["gold_T4"] = case("gold", 0, GL * CL^2 / (6.6743e-8 * 1.98847e33), 1.0,
                    2.4e22 / 3.0856775814913673e21, 0.9,
                    ["scene gold", "gold_A 100000.0", "gold_alpha 0.0", "gold_h $(repr(10 / 3))",
                     "gold_l0 1.0"])
r["ring"] = case("ring", 1, 1.3e10, 1.0e24, 16800.0, 0.9,
                 ["scene ring", "gamma $(repr(13 / 9))", "rat_low 1.0", "rat_high 1.0", "sigma_cut 1.0"])
r["pass"] = r["gold_T4"]["pass"] && r["ring"]["pass"]
mkpath(dirname(OUT))
write_json(OUT, r)
exit(r["pass"] ? 0 : 1)
