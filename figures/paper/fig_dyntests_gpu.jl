# fig_dyntests_gpu.jl — the dynamical-spacetime march of fig_dyntests.jl on the CPU threads
# and on a CUDA device: the moving Kerr hole and the orbiting binary of fig_dyntests.jl,
# 64² camera rays over the same field of view at the production step constant C = 80,
# traced by `trace` and `shadow` with `backend = :cpu` and `backend = :gpu`. Recorded:
# rays with equal step counts, the largest state difference along the rays (relative to
# max(1, |y|) componentwise), the largest invariant drift (|ΔE'|, |ΔL'|, |ΔQ'| or |ΔJ|) on
# each backend's states, equal shadow fates and step counts, and the wall times. No figure;
# the numbers enter the text of Sec. dyntests (CPU against GPU).
# Writes figures/paper/out/dyntests/gpu.json (BLP_DYNTESTS_OUT).
# Run inside a Slurm GPU job: results/blp/gpu/dyntests_gpu.sbatch
# (julia -t 4 --project=results/blp/gpu/env_main figures/paper/fig_dyntests_gpu.jl)
using CUDA
include(joinpath(@__DIR__, "fig_dyntests.jl"))

const NPIX_GPU = 64

function compare(name, st, inv)
    cfg = RunConfig(camera = CameraConfig(inclination_deg = INC, distance = DCAM, fov = FOV,
                                          resolution = NPIX_GPU),
                    C = C_PROD, max_steps = 100_000, escape_factor = ESCAPE / DCAM)
    plan = prepare(cfg; spacetime = st, matter = nothing)
    rec = Dict{String,Any}("npix" => NPIX_GPU, "C" => C_PROD)
    t0 = time(); tc = trace(plan; backend = :cpu); rec["trace_cpu_seconds"] = time() - t0
    t0 = time(); tg = trace(plan; backend = :gpu); rec["trace_gpu_seconds_cold"] = time() - t0
    t0 = time(); tg = trace(plan; backend = :gpu); rec["trace_gpu_seconds_warm"] = time() - t0
    n = length(tc.states)
    same = count(k -> length(tc.states[k]) == length(tg.states[k]), 1:n)
    gap = 0.0
    for k in 1:n, j in 1:min(length(tc.states[k]), length(tg.states[k]))
        a = tc.states[k][j]; b = tg.states[k][j]
        gap = max(gap, maximum(abs.(a - b) ./ max.(1.0, abs.(a))))
    end
    drift(states) = collect(reduce((a, b) -> max.(a, b),
                                   (abs.(inv(y) - inv(s[end])) for s in states for y in s)))
    rec["rays"] = n
    rec["equal_step_counts"] = same
    rec["max_state_gap"] = gap
    rec["bitwise_equal_rays"] = count(k -> tc.states[k] == tg.states[k], 1:n)
    rec["max_invariant_drift_cpu"] = drift(tc.states)
    rec["max_invariant_drift_gpu"] = drift(tg.states)
    t0 = time(); sc = shadow(plan; backend = :cpu); rec["shadow_cpu_seconds"] = time() - t0
    sg = shadow(plan; backend = :gpu)
    t0 = time(); sg = shadow(plan; backend = :gpu); rec["shadow_gpu_seconds_warm"] = time() - t0
    rec["shadow_fates_cpu"] = [count(==(f), sc.fate) for f in (1, 2, 0)]
    rec["shadow_equal_fates"] = count(sc.fate .== sg.fate)
    rec["shadow_equal_steps"] = count(sc.steps .== sg.steps)
    @printf("%-6s trace: %d/%d rays with equal step counts, %d bitwise equal, max state gap %.1e; invariant drift cpu %s gpu %s; cpu %.1f s, gpu %.1f s (%.1f s warm)\n",
            name, same, n, rec["bitwise_equal_rays"], gap, join((@sprintf("%.1e", v) for v in rec["max_invariant_drift_cpu"]), "/"),
            join((@sprintf("%.1e", v) for v in rec["max_invariant_drift_gpu"]), "/"), rec["trace_cpu_seconds"], rec["trace_gpu_seconds_cold"],
            rec["trace_gpu_seconds_warm"])
    @printf("%-6s shadow: fates (captured, escaped, step-limited) %s; equal fates %d/%d, equal step counts %d/%d\n",
            name, string(rec["shadow_fates_cpu"]), rec["shadow_equal_fates"], n, rec["shadow_equal_steps"], n)
    flush(stdout)
    return rec
end

function main_gpu()
    println("CUDA functional: ", CUDA.functional())
    CUDA.functional() && println(CUDA.name(CUDA.device()))
    mkpath(OUT)
    rec = Dict{String,Any}("git_sha" => GIT_SHA, "threads" => Threads.nthreads(),
                           "device" => CUDA.functional() ? CUDA.name(CUDA.device()) : "none")
    for (name, st, inv) in (("binary", binary(), binary_invariants),
                            ("kerr", moving_kerr(), kerr_invariants))
        rec[name] = try
            compare(name, st, inv)
        catch err
            msg = sprint(showerror, err)
            println("$name FAILED: ", first(msg, 2000))
            Dict{String,Any}("error" => first(msg, 2000))
        end
    end
    open(joinpath(OUT, "gpu.json"), "w") do io
        write(io, json_str(rec), "\n")
    end
    println("wrote ", joinpath(OUT, "gpu.json"))
end

main_gpu()
