# The BENCH / PHASE / SKIP record format of bench/bench_multigpu.jl (results/blp/multigpu/
# dag.md, MG01): formatting and parsing round-trip, free-text SKIP reasons, and the
# rejection of malformed records.
using Test
include(joinpath(@__DIR__, "..", "bench", "bench_lines.jl"))

@testset "bench_lines: BENCH / PHASE / SKIP records" begin
    b = bench_line("BENCH", :leg => "fused_768", :backend => "cuda", :ndev => 2,
                   :items => 589824, :wall_s => 0.4812345678, :first_s => 3.25,
                   :rate => 1.2257e6, :unit => "rays/s", :speedup => 1.9123, :equal => true,
                   :commit => "e291fbc")
    @test startswith(b, "BENCH leg=fused_768 backend=cuda ndev=2 items=589824 wall_s=0.481235")
    r = parse_bench_line(b)
    @test r.kind == "BENCH" && r.leg == "fused_768" && r.backend == "cuda"
    @test r.ndev === 2 && r.items === 589824
    @test r.wall_s ≈ 0.481235 && r.rate ≈ 1.2257e6 && r.speedup ≈ 1.9123
    @test r.equal === true && r.unit == "rays/s" && r.commit == "e291fbc"

    p = parse_bench_line(bench_line("PHASE", :leg => "amr_pol_384", :backend => "cuda",
                                    :ndev => 2, :device => 1, :phase => "kernel",
                                    :seconds => 0.125))
    @test p.kind == "PHASE" && p.device == "1" && p.phase == "kernel" && p.seconds == 0.125

    s = parse_bench_line("SKIP leg=mc_thin reason=CUDA not functional (NVML error)")
    @test s.kind == "SKIP" && s.leg == "mc_thin"
    @test s.reason == "CUDA not functional (NVML error)"

    text = """
    # bench_multigpu backend=cuda commit=e291fbc
    $b
    unrelated output
    SKIP leg=amr_driver_pol reason=missing snapshot /x.athdf
    """
    recs = parse_bench(text)
    @test [x.kind for x in recs] == ["BENCH", "SKIP"]
    @test recs[2].reason == "missing snapshot /x.athdf"
    @test parse_bench_line("BENCHMARK leg=x") === nothing

    @test_throws ErrorException bench_line("TIME", :leg => "x")
    @test_throws ErrorException bench_line("BENCH", :leg => "a b")
    @test_throws ErrorException parse_bench_line("BENCH leg=x junk")
end
