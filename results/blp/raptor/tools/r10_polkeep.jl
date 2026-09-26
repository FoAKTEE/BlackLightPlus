# r10_polkeep.jl — test of RAPTOR's polarized readout for radiation whose polarized
# invariant stays at or below the upstream activation threshold (I_pol,inv <= 1e-100 in
# pol_integration_step): the Gold et al. 2020 T3 emitter (unpolarized emission, j_Q = j_V
# = 0, no rotativities) on 64 pixels at i = 60, through the polarized (libraptor_rt_pol)
# and the unpolarized (libraptor_rt_unpol) single-pixel libraries of the fork.
# Upstream RAPTOR reads such a pixel out as I = Q = U = V = 0 (POLARIZATION_ACTIVE is
# never set); with POL_KEEP_I the polarized transfer returns the transported I.
# Usage: julia --project=. r10_polkeep.jl <bindir>
# Exit 0 iff every pixel with unpolarized I > 0 has |I_pol / I_unpol - 1| <= 1e-10 and
# Q = U = V = 0 in the polarized build.

using Printf
const BINDIR = ARGS[1]
const TMP = mktempdir()
const FOV = 30.0
const NU = 230.0e9
const GOLD_L = 6.0e11
const GOLD_D = 2.4e22
const MSUN_R = 1.98847e33
const G_R = 6.6743e-8
const CL = 2.99792458e10
const KPC = 3.0856775814913673e21

function wfile(name, lines)
    p = joinpath(TMP, name)
    open(io -> foreach(l -> println(io, l), lines), p, "w")
    return p
end

function pixels(lib, coords)
    h = Libc.Libdl.dlopen(joinpath(BINDIR, lib))
    setup = Libc.Libdl.dlsym(h, :raptor_rt_setup); pix = Libc.Libdl.dlsym(h, :raptor_rt_pixel)
    mi = wfile("model.in", [
        "MBH (Msun) $(repr(GOLD_L * CL^2 / (G_R * MSUN_R)))", "DISTANCE (kpc) $(repr(GOLD_D / KPC))",
        "M_UNIT (g) 1.0", "R_LOW (-) 1", "R_HIGH (-) 1", "INCLINATION (deg) 60.0",
        "IMG_WIDTH (pixels) 10", "IMG_HEIGHT (pixels) 10", "CAM_SIZE_X (Rg) $FOV",
        "CAM_SIZE_Y (Rg) $FOV", "FREQS_PER_DEC (-) 1", "FREQ_MIN (Hz) $(repr(NU))",
        "STEPSIZE (-) 0.01", "MAX_LEVEL (-) 1"])
    ENV["RAPTOR_BBH"] = wfile("bbh.in", ["nholes 1", "mass 1.0", "spin 0.9", "center1 0.0 0.0 0.0",
                                         "C 80.0", "nmax 8000", "escape 1100.0", "horizon_eps 0.001",
                                         "capture_redshift 100.0", "cam_dist 1000.0"])
    ENV["RAPTOR_SCENE"] = wfile("scene.in", ["scene gold", "gold_A 0.0", "gold_alpha 0.0",
                                             "gold_h $(repr(10 / 3))", "gold_l0 1.0", "gold_beta 2.5",
                                             "gold_Cn0 3.0e-18", "gold_nup $(repr(NU))"])
    redirect_stdout(devnull) do
        ccall(setup, Cvoid, (Cstring,), mi)
    end
    out = zeros(4, length(coords))
    for (i, c) in enumerate(coords)
        s = zeros(4)
        ccall(pix, Cint, (Cdouble, Cdouble, Ptr{Float64}), c[1], c[2], s)
        out[:, i] .= s
    end
    return out
end

coords = [(-FOV / 2 + (i - 0.5) * FOV / 8, -FOV / 2 + (j - 0.5) * FOV / 8) for i in 1:8 for j in 1:8]
P = pixels("libraptor_rt_pol.so", coords)
U = pixels("libraptor_rt_unpol.so", coords)
lit = [k for k in eachindex(coords) if U[1, k] > 0]
rel = maximum(abs(P[1, k] / U[1, k] - 1) for k in lit)
qv = maximum(abs.(P[2:4, :]))
ok = !isempty(lit) && rel <= 1e-10 && qv == 0
@printf("%d of %d pixels lit; max |I_pol / I_unpol - 1| = %.3e; max |Q, U, V| = %.1e; zero-I polarized pixels %d | %s\n",
        length(lit), length(coords), rel, qv, count(k -> P[1, k] == 0, lit), ok ? "PASS" : "FAIL")
exit(ok ? 0 : 1)
