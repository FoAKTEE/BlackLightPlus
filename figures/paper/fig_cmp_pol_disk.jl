# fig_cmp_pol (thin-disk part): the polarized Novikov–Thorne thin-disk image of Prather et
# al. (2023, Sec. 3.2) in BlackLightPlus, ipole and ipole-IL.
#
# BlackLightPlus: results/blp/rt_compare/polarized/thindisk/blp_thindisk.jl at the march
# constants C = 40, 80, 160, 320, 640 on the pixel centres, and at C = 80, 640 with every ray
# shifted by -0.01 pixel in α (the ray offset of both ipole codes). ipole
# (Moscibrodzka & Gammie 2018, v2.0, model_NT.c) at its step parameter eps = 0.005 (its
# default for this problem), 0.0025, 0.00125: run_ipole_nt.sh. ipole-IL (AFD-Illinois,
# model thin_disk, deck tests/thin_disk/thin_disk.par) at eps = 0.01 (default), 0.005,
# 0.0025: run_ipoleil_nt.sh.
#
# Writes figures/paper/out/cmp_pol/thindisk/<run>/{I,Q,U,V}.npy in Jy per pixel, indexed
# [α pixel, β pixel] with α increasing to the west and β to the north (the projected spin
# axis), Stokes in the IAU convention (Q > 0 north-south, U > 0 north-east). Mappings: the ipole
# dump columns (i, j, I, -Q, -U, V) are already IAU and indexed [α, β]; the ipole-IL
# `pol` dataset (C order [i][j][s]) is IAU ("evpa_0 = N") and indexed [α, β].
#
# Run: julia --project=. -t 16 figures/paper/fig_cmp_pol_disk.jl
using BlackLightPlus, HDF5

const ROOT = dirname(dirname(@__DIR__))
const RC = joinpath(ROOT, "results", "blp", "rt_compare")
const TD = joinpath(RC, "polarized", "thindisk")
const BUILD = joinpath(RC, "build")
const OUT = joinpath(ROOT, "figures", "paper", "out", "cmp_pol", "thindisk")
const NTHREADS = string(min(Threads.nthreads(), 16))
const IPOLE_SRC = joinpath(BUILD, "ipole")        # copy of ref-code ipole (Moscibrodzka)
const IPOLEIL_SRC = joinpath(BUILD, "ipole-il")   # clone of AFD-Illinois ipole at 7f7a482

function blp_runs()
    script = joinpath(TD, "blp_thindisk.jl")
    for C in (40, 80, 160, 320, 640)
        run(`julia --project=$ROOT -t $NTHREADS $script $(joinpath(OUT, "blp_C$C")) $C 0`)
    end
    for C in (80, 640)
        run(`julia --project=$ROOT -t $NTHREADS $script $(joinpath(OUT, "blp_C$(C)_ipoleoff")) $C -0.01`)
    end
end

function ipole_runs()
    for eps in ("0.005", "0.0025", "0.00125")
        d = joinpath(OUT, "ipole_eps$eps")
        run(addenv(`$(joinpath(TD, "run_ipole_nt.sh")) $IPOLE_SRC $(joinpath(BUILD, "ipole_nt_eps$eps")) $eps $d`,
                   "OMP_NUM_THREADS" => NTHREADS))
        rows = [parse.(Float64, split(l)) for l in eachline(joinpath(d, "ipole.dat"))]
        n = 80
        for (k, c) in zip(4:7, ("I", "Q", "U", "V"))
            m = zeros(n, n)
            for r in rows
                m[Int(r[1]) + 1, Int(r[2]) + 1] = r[k]
            end
            write_npy(joinpath(d, "$c.npy"), m)
        end
    end
end

function ipoleil_runs()
    for eps in ("0.01", "0.005", "0.0025")
        d = joinpath(OUT, "ipoleil_eps$eps")
        run(addenv(`$(joinpath(TD, "run_ipoleil_nt.sh")) $IPOLEIL_SRC $(joinpath(BUILD, "ipoleil_nt")) $eps $d`,
                   "OMP_NUM_THREADS" => NTHREADS))
        h = h5open(joinpath(d, "image.h5"))
        pol = read(h["pol"]); scale = read(h["header/scale"]); close(h)
        for (k, c) in enumerate(("I", "Q", "U", "V"))
            write_npy(joinpath(d, "$c.npy"), permutedims(pol[k, :, :]) .* scale)
        end
    end
end

mkpath(OUT)
blp_runs()
ipole_runs()
ipoleil_runs()
println("wrote ", OUT)
