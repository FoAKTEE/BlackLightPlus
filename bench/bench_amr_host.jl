# bench_amr_host.jl — host time of the AMR driver (render_amr) on a synthetic deck
# (results/blp/multigpu/design.md §0.1, dag.md MG10): a Stokes closure that costs almost
# nothing (a sharp analytic ring), base 384, block 8, 2 levels. The host time is the
# render_amr wall time minus the time spent inside the closure: block coordinates,
# painting, the refinement indicator and the canvas. Best of BLP_REPEAT (9) warm runs,
# each after a full GC (the previous run's garbage is not charged to the next); one BENCH
# line (bench_lines.jl format), `variant` = Julia thread count.
#   julia --project=. -t 1  bench/bench_amr_host.jl
#   julia --project=. -t 16 bench/bench_amr_host.jl
# It uses only the public render_amr, so it also times other commits from a worktree.
using BlackLightPlus
using BlackLightPlus.Inputs: RunConfig, CameraConfig
include(joinpath(@__DIR__, "bench_lines.jl"))

const REPEAT = parse(Int, get(ENV, "BLP_REPEAT", "9"))
const INSIDE = Ref(0.0)                       # seconds spent in the closure this run

function ring_stokes(coords)
    t = time_ns()
    S = Matrix{Float64}(undef, 4, length(coords))
    Threads.@threads for k in eachindex(coords)
        α, β = coords[k]
        I = exp(-((hypot(α, 0.8β) - 5.0) / 0.3)^2) + 0.01
        S[1, k] = I; S[2, k] = 0.3I; S[3, k] = 0.2I; S[4, k] = 0.01I
    end
    INSIDE[] += (time_ns() - t) / 1e9
    return S
end

function main()
    cfg = RunConfig(bh_a = 0.9, camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                                       fov = 24.0, resolution = 384))
    run() = (GC.gc(); INSIDE[] = 0.0; t = time_ns();
             r = render_amr(cfg, ""; block = 8, max_level = 2, stokes = ring_stokes);
             ((time_ns() - t) / 1e9 - INSIDE[], r.nrays))
    run()                                     # compile
    host, nrays = minimum(run() for _ in 1:REPEAT)
    commit = try readchomp(`git -C $(dirname(Base.active_project())) rev-parse --short HEAD`) catch; "unknown" end
    println(bench_line("BENCH", :leg => "amr_host", :backend => "cpu", :ndev => 0,
                       :items => nrays, :wall_s => host, :first_s => NaN, :rate => nrays / host,
                       :unit => "rays/s", :variant => "threads$(Threads.nthreads())",
                       :commit => commit))
end

main()
