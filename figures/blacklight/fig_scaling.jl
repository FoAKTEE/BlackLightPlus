# fig_scaling.jl — performance of the fiducial deck on CPU threads (paper figures 16-17).
# Mode `strong` (run once per thread count, julia -t N): the 256² Stokes I image rendered
# twice, the second timed, reported as rays per second per thread. Mode `resolution`
# (julia -t 4): one image at 64², 128², 256², 512² and 1024², and ten images at once
# (`variants` over ten mass scales) at each, both timed after a warm-up. Each run appends
# one JSON line to out/blacklight/scaling/results.jsonl. Driver: run_scaling.sh.
include(joinpath(@__DIR__, "common.jl"))

function main()
    out = outdir("scaling")
    mode = ARGS[1]
    snap = AthenaKSnapshot(dump_path())
    kerr = KerrSchildCartesian(BL_SPIN)
    f = open(joinpath(out, "results.jsonl"), "a")
    if mode == "strong"
        plan = prepare(fiducial_config(; res = 256); spacetime = kerr, matter = snap, mode = StokesI())
        render(plan; backend = :cpu)
        t0 = time(); render(plan; backend = :cpu); dt = time() - t0
        println(f, json_str(Dict("mode" => "strong", "threads" => Threads.nthreads(), "res" => 256,
                                 "seconds" => dt, "rays_per_s" => 256^2 / dt, "host_cpus" => Sys.CPU_THREADS)))
        logf("strong: %d threads, 256^2 in %.2f s, %.0f rays/s/thread\n", Threads.nthreads(), dt, 256^2 / dt / Threads.nthreads())
    elseif mode == "resolution"
        render(prepare(fiducial_config(; res = 64); spacetime = kerr, matter = snap, mode = StokesI()); backend = :cpu)
        L3 = BL_L^3
        ladder = [mass_scale() * 10.0^e for e in range(-0.5, 0.5; length = 10)]
        for res in (64, 128, 256, 512, 1024)
            plan = prepare(fiducial_config(; res = res); spacetime = kerr, matter = snap, mode = StokesI())
            t0 = time(); render(plan; backend = :cpu); dt1 = time() - t0
            t0 = time(); variants(plan, ladder; backend = :cpu); dt10 = time() - t0
            println(f, json_str(Dict("mode" => "resolution", "threads" => Threads.nthreads(), "res" => res,
                                     "seconds_one" => dt1, "seconds_ten" => dt10)))
            logf("resolution %d: one image %.2f s, ten images %.2f s\n", res, dt1, dt10)
            flush(f)
        end
    end
    close(f)
end
main()
