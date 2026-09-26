# r10_blp.jl — BlackLightPlus uniform images of the i = 90 AMR-figure deck (dag.md R10;
# figures/fig_bbh_i90_pol_amr.jl): s455 (gamma = 13/9) on the static SuperposedKerrSchild
# binary m1 = m2 = 0.5, a = 0, centres +-12.5 (cos phi0, sin phi0, 0), phi0 = -0.2233022;
# 230 GHz; i = 90; distance 1000 M; C = 80; max_steps 2500; m_bh = 6.5e9 x 1.989e33 g
# (total); mass_scale 1e24 g; PlasmaConfig defaults (R_low 1, R_high 10, mu 0.5,
# sigma_cut 1). The rays are traced and marched exactly as the figure's closure
# (camera_rays_at -> trace_rays_sampled in chunks -> Radiation.polarized_march).
#
# Per (fov, npix) it writes <out>/blp_i90_C<C>_n<npix>[_fov<fov>]/{I,Q,U,V,class}.npy
# (M[i_beta, i_alpha], cgs intensity) and record.json (walls, rays/s, classes), and
# <out>/bbh.in (the deck's binary in RAPTOR's format).
#
# Env R10_C, R10_NMAX override the deck step constant and budget (attribution legs).
# Usage: julia --project=. -t N r10_blp.jl <out> <npix,...> [fov=40]

using StaticArrays, Printf, Dates, Statistics
using BlackLightPlus
const BLP = BlackLightPlus
const RAD = BLP.Radiation
const SIM = BLP.Simulation
include(joinpath(@__DIR__, "jsonw.jl"))

const OUT = ARGS[1]
const NPIXS = parse.(Int, split(ARGS[2], ","))
const FOV = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 40.0
const BIN = get(ENV, "BLP_BBH_BIN",
    "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const PHI0 = -0.2233022
const GAMMA = 13.0 / 9.0
const CVAL = parse(Float64, get(ENV, "R10_C", "80"))
const INC = 90.0
const DIST = 1000.0
const NU = 2.3e11
const MAXSTEPS = parse(Int, get(ENV, "R10_NMAX", "2500"))
const M_BH = 6.5e9 * 1.989e33
const MASS_SCALE = 1.0e24
const ESCAPE = 1.1 * DIST
const CHUNK = 65536

const C1 = 12.5 .* (cos(PHI0), sin(PHI0), 0.0)
const C2 = (-C1[1], -C1[2], 0.0)
const ST = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [C1, C2])
const CFG = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE,
                      frequency_hz = NU, polarized = true,
                      plasma = PlasmaConfig(gamma = GAMMA), max_steps = MAXSTEPS)
const UNITS = SIM.units(M_BH, MASS_SCALE)
const SCC = BLP._prepared_scc(CFG, UNITS)

function end_class(y)
    all(isfinite, y) || return 5
    x = SVector(y[1], y[2], y[3], y[4])
    if BLP.Geodesics.superposed_control(ST, y, CVAL)[1]
        d1 = ks_radius_hole(ST, x, 1) - horizon_radius_hole(ST, 1)
        d2 = ks_radius_hole(ST, x, 2) - horizon_radius_hole(ST, 2)
        return d1 <= d2 ? 1 : 2
    end
    sqrt(y[2]^2 + y[3]^2 + y[4]^2) > ESCAPE && return 3
    return 4
end

"Stokes (4 x n), classes and sample counts at image coordinates (the figure's closure)."
function stokes(cam, up, sampler, coords)
    n = length(coords)
    S = zeros(4, n); cls = zeros(Int8, n); ns = zeros(Int32, n)
    ttr = 0.0; tpol = 0.0
    for lo in 1:CHUNK:n
        hi = min(n, lo + CHUNK - 1)
        t0 = time()
        rays = camera_rays_at(ST, cam, coords[lo:hi])
        states, dls = trace_rays_sampled(ST, rays; C = CVAL, nmax = MAXSTEPS, escape = ESCAPE)
        t1 = time()
        Threads.@threads :dynamic for k in eachindex(rays)
            i = lo + k - 1
            ys = states[k]
            cls[i] = end_class(ys[1]); ns[i] = length(ys)
            s4 = RAD.polarized_march(ST, SCC, sampler, ys, dls[k], NU, UNITS.L_unit; cam_up = up)
            S[1, i] = s4[1]; S[2, i] = s4[2]; S[3, i] = s4[3]; S[4, i] = s4[4]
        end
        ttr += t1 - t0; tpol += time() - t1
    end
    return S, cls, ns, ttr, tpol
end

function main()
    mkpath(OUT)
    sha = try readchomp(`git -C $(dirname(dirname(dirname(dirname(@__DIR__))))) rev-parse HEAD`) catch; "unknown" end
    t0 = time()
    snap = athenak_bin_snapshot(BIN; gamma = GAMMA)
    load_s = time() - t0
    sampler = x -> begin
        s = SIM.sample_primitives(snap, x)
        (rho = s.rho, pgas = SIM.pgas_from_eint(s.eint, GAMMA), uu = s.uu, bb = s.bb, inside = s.inside)
    end
    open(joinpath(OUT, "bbh.in"), "w") do io
        println(io, "nholes 2\nmass 0.5 0.5\nspin 0.0 0.0")
        println(io, "center1 $(repr(C1[1])) $(repr(C1[2])) $(repr(C1[3]))")
        println(io, "center2 $(repr(C2[1])) $(repr(C2[2])) $(repr(C2[3]))")
        println(io, "C $(repr(CVAL))\nnmax $(MAXSTEPS)\nescape $(repr(ESCAPE))")
        println(io, "horizon_eps 0.001\ncapture_redshift 100.0\ncam_dist $(repr(DIST))")
    end
    @printf("BLP %s | threads %d | load %.2f s | maxrss %.1f GB\n", sha, Threads.nthreads(), load_s, Sys.maxrss() / 2^30)
    cam24 = PlaneParallelCamera(INC, DIST, FOV, 24)
    tw = time()
    stokes(cam24, BLP._camera_up(cam24), sampler, [(0.0, 0.0), (1.0, 1.0)])      # compilation
    jit_s = time() - tw
    for npix in NPIXS
        cam = PlaneParallelCamera(INC, DIST, FOV, npix)
        up = BLP._camera_up(cam)
        ps = FOV / npix
        coords = [(-FOV / 2 + (i - 0.5) * ps, -FOV / 2 + (j - 0.5) * ps) for i in 1:npix for j in 1:npix]
        GC.gc()
        tr = time()
        S, cls, ns, ttr, tpol = stokes(cam, up, sampler, coords)
        wall = time() - tr
        tag = FOV == 40.0 ? "" : @sprintf("_fov%g", FOV)
        dir = joinpath(OUT, "blp_i90_C$(round(Int, CVAL))_n$(npix)$(tag)")
        mkpath(dir)
        for (c, nm) in enumerate(("I", "Q", "U", "V"))
            write_npy(joinpath(dir, "$(nm).npy"), reshape(S[c, :], npix, npix))
        end
        write_npy(joinpath(dir, "class.npy"), reshape(Float64.(cls), npix, npix))
        I = S[1, :]; P = hypot.(S[2, :], S[3, :])
        rec = Dict("code" => "BlackLightPlus", "git_sha" => sha, "snapshot" => BIN, "npix" => npix,
                   "fov_M" => FOV, "inclination_deg" => INC, "C" => CVAL, "nmax" => MAXSTEPS,
                   "distance_M" => DIST, "nu_Hz" => NU, "M_bh_g" => M_BH, "mass_scale_g" => MASS_SCALE,
                   "gamma" => GAMMA, "R_low" => 1.0, "R_high" => 10.0, "sigma_cut" => 1.0,
                   "centers" => [collect(C1), collect(C2)], "phi0_rad" => PHI0,
                   "threads" => Threads.nthreads(), "load_s" => load_s, "jit_warmup_s" => jit_s,
                   "image_wall_s" => wall, "trace_s" => ttr, "transfer_s" => tpol,
                   "rays" => npix^2, "rays_per_s" => npix^2 / wall, "samples" => sum(Int.(ns)),
                   "area_flux_I" => sum(I) * ps^2, "LP" => sum(P) / sum(I),
                   "classes_cap1_cap2_esc_nmax_nonfinite" => [count(==(k), cls) for k in 1:5],
                   "maxrss_GB" => Sys.maxrss() / 2^30, "when" => string(Dates.now()))
        write_json(joinpath(dir, "record.json"), rec)
        @printf("n=%d fov=%g: %.1f s (trace %.1f, transfer %.1f; %.0f rays/s) | area flux %.6e | LP %.4f | classes %s\n",
                npix, FOV, wall, ttr, tpol, npix^2 / wall, rec["area_flux_I"], rec["LP"],
                string(rec["classes_cap1_cap2_esc_nmax_nonfinite"]))
        flush(stdout)
    end
end
main()
