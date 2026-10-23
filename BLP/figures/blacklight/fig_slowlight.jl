# fig_slowlight.jl — fast light against slow light (paper figure 7): the fiducial deck with
# the camera centre moved in to r = 25 M. Fast light: the dump at t = 78540 M (file 00998)
# held fixed. Slow light: the seven dumps 00993 … 00999 (t = 78390 … 78570 M, cadence
# 30 M) sampled at each ray's own emission time t_obs + x⁰ with t_obs = 78570 M, the
# bracketing dumps blended linearly (SeriesProvider). The paper's series had 160 dumps
# 1 M apart; here the cadence is 30 M, so the blend interpolates the fluid over 30 M.
# Writes out/blacklight/slowlight/{fast_I.npy, slow_I.npy, record.json}.
# Run from the repository root:   julia -t 32 --project=BLP BLP/figures/blacklight/fig_slowlight.jl
include(joinpath(@__DIR__, "common.jl"))
using BLP.Data: SnapshotSeries, SeriesProvider

dump_time(n) = 54030.0 + 30.0 * (n - 181)

function main()
    out = outdir("slowlight")
    kerr = KerrSchildCartesian(BL_SPIN)
    cfg = fiducial_config(; dcam = 25.0, max_steps = 6000)
    rec = Dict{String,Any}()
    # fast light
    t0 = time()
    fast = render(prepare(cfg; spacetime = kerr, matter = AthenaKSnapshot(dump_path(998)), mode = StokesI()); backend = :cpu)
    write_npy(joinpath(out, "fast_I.npy"), fast.I)
    rec["fast"] = Dict("dump" => 998, "time" => dump_time(998), "flux_jy" => flux_jy(fast.I, BL_FOV, BL_RES), "seconds" => time() - t0)
    logf("fast light: %.3f Jy in %.0f s\n", rec["fast"]["flux_jy"], rec["fast"]["seconds"])
    # slow light
    ns = collect(993:999)
    series = SnapshotSeries([dump_path(n) for n in ns], [dump_time(n) for n in ns]; maxslots = 8, t_obs = dump_time(999))
    t0 = time()
    slow = render(prepare(cfg; spacetime = kerr, matter = SeriesProvider(series), mode = StokesI()); backend = :cpu)
    write_npy(joinpath(out, "slow_I.npy"), slow.I)
    rec["slow"] = Dict("dumps" => ns, "times" => dump_time.(ns), "t_obs" => dump_time(999),
                       "flux_jy" => flux_jy(slow.I, BL_FOV, BL_RES), "seconds" => time() - t0)
    logf("slow light: %.3f Jy in %.0f s\n", rec["slow"]["flux_jy"], rec["slow"]["seconds"])
    # the elapsed coordinate time of the rays (for the span of dumps needed)
    traj = trace(prepare(cfg; spacetime = kerr, matter = nothing, mode = StokesI()); backend = :cpu)
    span = maximum(abs(s[1][1]) for s in traj.states)
    rec["max_elapsed_time"] = span
    logf("longest geodesic time span %.1f M\n", span)
    write_json(joinpath(out, "record.json"), record("slowlight"; result = rec))
end
main()
