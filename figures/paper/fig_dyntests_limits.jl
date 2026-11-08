# fig_dyntests_limits.jl — two limits of the time-dependent march, checked ray by ray and
# image by image (no figure; the numbers enter the text of Sec. dyntests).
# (1) Zero orbital frequency: the binary of fig_dyntests.jl with ω = 0 (holes at rest at
#     the orbital phase of t = -d, boosted terms) is the static superposition
#     `SuperposedKerrSchild` at the same centres; the time-dependent march (geodesic force
#     with the AD time column) and the stationary march (dp_t/dλ = 0 by construction) are
#     compared on the 64² camera of fig_dyntests.jl at C = 80: fates, step counts, and the
#     largest state difference relative to max(1, |y|) componentwise.
# (2) Slow light on a static fluid series: the AthenaK torus snapshot 00999 repeated at
#     three times 30 apart (`synthetic_series`, the identity series) on the moving binary of
#     results/blp/gpu/moving_binary.jl (masses 1/2, no spin, separation 8, Keplerian),
#     rendered in Stokes I and IQUV at 48² pixels, with the plasma and M87* scaling of
#     fig_pol_hires.jl (m87_scaling.jl), by the snapshot (fast light), the host
#     `SeriesProvider` and the two-frame `window` (slow light): the largest pixel difference
#     over the largest value, and whether the images are bitwise equal. Snapshot path:
#     BLP_TORUS (default the moving_binary.jl data directory).
# Writes figures/paper/out/dyntests/limits.json (BLP_DYNTESTS_OUT).
# Run from the repository root: julia -t 24 --project=. figures/paper/fig_dyntests_limits.jl
include(joinpath(@__DIR__, "fig_dyntests.jl"))
using BlackLightPlus.Spacetime: SuperposedKerrSchild
using BlackLightPlus.Data: synthetic_series, window
include(joinpath(@__DIR__, "m87_scaling.jl"))
using .M87Scaling

const NPIX_LIM = 64
const TORUS = get(ENV, "BLP_TORUS",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")

function zero_frequency()
    t0 = time()
    orbit0 = CircularOrbit(M_BIN, SEP; omega = 0.0, phase = PHASE)
    spins = (CHI[1] * M_BIN[1], CHI[2] * M_BIN[2])
    moving = BoostedSuperposedKerrSchild(collect(M_BIN), [(0.0, 0.0, spins[1]), (0.0, 0.0, spins[2])], orbit0)
    (s1, _), (s2, _) = orbit0(0.0)
    static = SuperposedKerrSchild(collect(M_BIN), collect(spins), [Tuple(s1), Tuple(s2)])
    cam = PlaneParallelCamera(inc_deg = INC, d = DCAM, fov = FOV, npix = NPIX_LIM)
    fr = CameraFrame(cam); cen = pixel_centres(cam)
    n = NPIX_LIM^2
    res = Vector{NTuple{5,Float64}}(undef, n)
    Threads.@threads :dynamic for k in 1:n
        α, β = pixel_coords(cen, NPIX_LIM, k)
        ym, _ = camera_ray(moving, fr, α, β)
        ys, _ = camera_ray(static, fr, α, β)
        rm = march_ray(moving, binary_invariants, ym, C_PROD; keep = true)
        rs = march_ray(static, binary_invariants, ys, C_PROD; keep = true)
        gap = 0.0
        for j in 1:min(length(rm.states), length(rs.states))
            a = rm.states[j]; b = rs.states[j]
            gap = max(gap, maximum(abs.(a - b) ./ max.(1.0, abs.(a))))
        end
        res[k] = (rm.fate, rs.fate, rm.n, rs.n, gap)
    end
    rec = Dict{String,Any}(
        "rays" => n, "C" => C_PROD, "camera_rays_equal" => count(k -> begin
            α, β = pixel_coords(cen, NPIX_LIM, k)
            camera_ray(moving, fr, α, β)[1] == camera_ray(static, fr, α, β)[1]
        end, 1:n),
        "equal_fates" => count(r -> r[1] == r[2], res), "equal_steps" => count(r -> r[3] == r[4], res),
        "max_state_gap" => maximum(r -> r[5], res),
        "max_state_gap_escaped" => maximum(r -> r[1] == 2 ? r[5] : 0.0, res),
        "median_state_gap" => sort([r[5] for r in res])[cld(n, 2)],
        "captured" => count(r -> r[1] == 1, res), "wall_seconds" => time() - t0)
    @printf("omega = 0: camera rays equal %d/%d; fates equal %d/%d, step counts equal %d/%d; max state gap %.1e (escaped %.1e, median %.1e); %.1f s\n",
            rec["camera_rays_equal"], n, rec["equal_fates"], n, rec["equal_steps"], n, rec["max_state_gap"],
            rec["max_state_gap_escaped"], rec["median_state_gap"], rec["wall_seconds"])
    return rec
end

function static_series()
    t0 = time()
    cfg = RunConfig(m_bh_cgs = M87_MASS_G, mass_scale_cgs = m87_density_scale("00999"),
                    frequency_hz = M87_FREQUENCY_HZ,
                    camera = CameraConfig(inclination_deg = 60.0, distance = 100.0, fov = 30.0, resolution = 48),
                    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13/9, use_p = true, sigma_cut = 1.0),
                    max_steps = 2500, C = 80.0, escape_factor = 1.1)
    moving = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], CircularOrbit((0.5, 0.5), 8.0))
    snap = AthenaKSnapshot(TORUS)
    times = [snap.time - 60.0, snap.time - 30.0, snap.time]
    series = synthetic_series(snap; times = times)
    win = window(series, 2)
    rec = Dict{String,Any}("snapshot" => TORUS, "times" => times, "npix" => 48)
    rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
    for (mname, mode) in (("StokesI", StokesI()), ("IQUV", IQUV()))
        fast = render(prepare(cfg; spacetime = moving, matter = snap, mode); backend = :cpu)
        for (pname, prov) in (("series", SeriesProvider(series)), ("window", win))
            slow = render(prepare(cfg; spacetime = moving, matter = prov, mode); backend = :cpu)
            ks = mode isa IQUV ? (:I, :Q, :U, :V) : (:I,)
            d = Dict(string(k) => rel(getproperty(slow, k), getproperty(fast, k)) for k in ks)
            eq = all(getproperty(slow, k) == getproperty(fast, k) for k in ks)
            rec["$(mname)_$(pname)"] = Dict{String,Any}("rel_diff" => d, "bitwise_equal" => eq,
                                                       "sum_I_fast" => sum(fast.I))
            @printf("static series, %s, %s: bitwise equal %s; max |slow - fast| / max |fast| %s\n", mname, pname, eq,
                    join(("$k $(@sprintf("%.1e", d[string(k)]))" for k in ks), ", "))
        end
    end
    rec["wall_seconds"] = time() - t0
    return rec
end

function main_limits()
    mkpath(OUT)
    rec = Dict{String,Any}("git_sha" => GIT_SHA, "threads" => Threads.nthreads(),
                           "zero_frequency" => zero_frequency(), "static_series" => static_series())
    open(joinpath(OUT, "limits.json"), "w") do io
        write(io, json_str(rec), "\n")
    end
    println("wrote ", joinpath(OUT, "limits.json"))
end

main_limits()
