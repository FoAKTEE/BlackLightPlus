# fig_scaling.jl — performance of the fiducial deck on CPU threads (paper figures 16-17).
# Mode `strong` (run once per thread count, julia -t N): the 256² Stokes I image rendered
# twice, the second timed, reported as rays per second per thread. Mode `resolution`
# (julia -t 4): the paper's ladder 8² … 1024² for three products, the polarized (IQUV)
# image, the Stokes I image, and ten Stokes I images at once (`variants` over ten mass
# scales); every product is compiled on an 8² plan first, then each point is rendered
# until RES_SECONDS have passed and at least RES_RUNS times, every wall time kept. Each
# run appends one JSON line to out/blacklight/scaling/results.jsonl. Driver:
# run_scaling.sh.
include(joinpath(@__DIR__, "common.jl"))

const RES_LADDER = (8, 16, 32, 64, 128, 256, 512, 1024)
const RES_RUNS = 3
const RES_SECONDS = 3.0

"the CPUs this process may run on (Linux `Cpus_allowed_list`), or \"\""
cpus_allowed() = try
    l = filter(s -> startswith(s, "Cpus_allowed_list"), readlines("/proc/self/status"))
    strip(split(only(l), ':')[2])
catch
    ""
end

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
        ladder = [mass_scale() * 10.0^e for e in range(-0.5, 0.5; length = 10)]
        # (name, transfer mode, images per call, the call); the call returns the first image
        products = (("iquv", IQUV(), 1, plan -> render(plan; backend = :cpu).I),
                    ("stokes_i", StokesI(), 1, plan -> render(plan; backend = :cpu).I),
                    ("ten_stokes_i", StokesI(), 10, plan -> variants(plan, ladder; backend = :cpu).images[1]))
        for (_, m, _, call) in products
            call(prepare(fiducial_config(; res = 8); spacetime = kerr, matter = snap, mode = m))
        end
        for res in RES_LADDER, (name, m, nimg, call) in products
            plan = prepare(fiducial_config(; res = res); spacetime = kerr, matter = snap, mode = m)
            secs = Float64[]
            I = nothing
            while length(secs) < RES_RUNS || sum(secs) < RES_SECONDS
                t0 = time_ns(); I = call(plan); push!(secs, (time_ns() - t0) / 1e9)
            end
            rate = nimg * res^2 / (sum(secs) / length(secs)) / Threads.nthreads()
            println(f, json_str(Dict("mode" => "resolution", "product" => name, "images" => nimg,
                                     "threads" => Threads.nthreads(), "res" => res, "seconds" => secs,
                                     "flux_jy" => flux_jy(I, BL_FOV, res), "code" => BL_GIT_SHA,
                                     "host_cpus" => Sys.CPU_THREADS, "cpus_allowed" => cpus_allowed(),
                                     "cpu" => Sys.cpu_info()[1].model, "date" => string(now()))))
            logf("resolution %4d %-12s %3d runs, mean %.4g s, %.0f rays/s/thread, %.3f Jy\n",
                 res, name, length(secs), sum(secs) / length(secs), rate, flux_jy(I, BL_FOV, res))
            flush(f)
        end
    end
    close(f)
end
main()
