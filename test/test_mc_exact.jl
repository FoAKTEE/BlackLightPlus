# Monte Carlo stream identity and exact accumulation (results/blp/multigpu/dag.md MG05,
# MG06), on the KernelAbstractions CPU backend:
#   - the exact accumulator (exact_sum.jl) gives the correctly rounded sum, bitwise
#     independent of order and of chunked merging;
#   - stream identities are injective over genealogies and separated by layer;
#   - the device MC kernels run on `HostDevice`s, reproduce the CPU drivers bit for bit,
#     and their spectra, diagnostics and child sets are bitwise independent of the device
#     count, the CPU thread (chunk) count, the child-buffer capacity and the run.
using Test, Random, StaticArrays, KernelAbstractions
using BlackLightPlus
using BlackLightPlus: SphereModel, ScatteringSphere, HotcrossTable, mc_spectrum,
                      mc_spectrum_scatter
using BlackLightPlus.Constants: SIGMA_THOMSON
using BlackLightPlus.MonteCarlo: NW, NT
const XM = BlackLightPlus.MonteCarlo

const _XE_SRC = joinpath(@__DIR__, "..", "src")
@isdefined(_fused_kernel!) || include(joinpath(_XE_SRC, "gpu_image.jl"))
@isdefined(_mc_device_shares) || include(joinpath(_XE_SRC, "montecarlo", "mc_gpu.jl"))

xe_bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
xe_bits(x::Real) = reinterpret(UInt64, Float64(x))
xe_hosts(nd) = [HostDevice(k) for k in 1:nd]

# Exact reference: BigFloat wide enough to hold any sum of Float64s exactly.
xe_ref(v) = setprecision(BigFloat, 2400) do
    Float64(sum(BigFloat.(v)))
end

function xe_sum(v)
    acc = XM.xsum_zeros(1)
    for x in v
        XM.xsum_add!(acc, x, 1)
    end
    return acc
end

# All bits of a ScatterResult: every order, the total and the diagnostics.
xe_scatter_bits(r) = vcat((xe_bits(o.sumw) for o in r.orders)...,
                          (xe_bits(o.sumwE) for o in r.orders)...,
                          xe_bits(r.total.sumw), xe_bits(r.total.sumwE),
                          xe_bits(r.w_primary), xe_bits(r.w_children),
                          xe_bits(r.w_first_order), xe_bits(r.scatter_loss))
xe_counts(r) = vcat((vec(o.count) for o in r.orders)..., r.n_children, r.n_primary)

@testset "MC exact accumulation and stream identity" begin

@testset "exact accumulator: order independence and correct rounding" begin
    rng = Xoshiro(20260923)
    # 10^6 positive weights over ~2^±150 (the span of the synchrotron weight table)
    w = exp.(50.0 .* randn(rng, 10^6))
    ref = xe_ref(w)
    b0 = XM.xsum_round(xe_sum(w), 1)
    @test b0 == ref                                  # correctly rounded exact sum
    @test abs(b0 - ref) / ref <= 1.0e-15
    same = true
    for _ in 1:100
        same &= xe_bits(XM.xsum_round(xe_sum(view(w, randperm(rng, length(w)))), 1)) ==
                xe_bits(b0)
    end
    @test same                                       # 100 random orders: identical bits
    # chunked accumulation merged by integer addition (threads / devices / stripes)
    for nch in (2, 7, 64)
        acc = XM.xsum_zeros(1)
        for c in 1:nch
            lo = div((c - 1) * length(w), nch) + 1; hi = div(c * length(w), nch)
            acc .+= xe_sum(view(w, lo:hi))
        end
        @test xe_bits(XM.xsum_round(acc, 1)) == xe_bits(b0)
    end

    # signed, subnormal, extreme and cancelling addends
    edge = [floatmax(), -floatmax(), 5.0e-324, 2.5e-308, -1.0, 1.0 + eps(), 1.0e300,
            -1.0e-300, 3.0, -3.0, nextfloat(0.0), prevfloat(floatmax()) / 3]
    mix = vcat(edge, ldexp.(randn(rng, 1000), rand(rng, -1000:1000)))
    @test XM.xsum_round(xe_sum(mix), 1) == xe_ref(mix)
    @test XM.xsum_round(xe_sum([1.0, 2.0^-80, -1.0]), 1) == 2.0^-80        # no absorption
    @test XM.xsum_round(xe_sum([5.0e-324, 5.0e-324]), 1) == 1.0e-323        # subnormal
    @test XM.xsum_round(xe_sum(Float64[]), 1) == 0.0
    @test isnan(XM.xsum_round(xe_sum([1.0, Inf]), 1))                       # non-finite flagged
    @test XM.xsum_round(xe_sum([floatmax(), floatmax()]), 1) == Inf         # overflow rounds up
    # round-half-even at 2^53
    @test XM.xsum_round(xe_sum([2.0^53, 1.0]), 1) == 2.0^53
    @test XM.xsum_round(xe_sum([2.0^53 + 2, 1.0]), 1) == 2.0^53 + 4
end

@testset "stream identity: injective, layer-separated, partition-free" begin
    # exhaustive genealogies of a small layout: gid 1:5, events 0:3, up to 3 layers
    L = XM.MCStreamLayout(5, 4, 3)
    @test (L.gid_bits, L.event_bits) == (3, 2)
    ids = Set{UInt128}()
    n = 0
    for gid in 1:5
        r0 = XM.mc_primary_rng(7, gid)
        push!(ids, XM.mc_stream_id(r0)); n += 1
        @test XM.mc_stream_layer(r0) == 0
        for e1 in 0:3
            r1 = XM.mc_child_rng(r0, e1, L)
            push!(ids, XM.mc_stream_id(r1)); n += 1
            @test XM.mc_stream_layer(r1) == 1
            for e2 in 0:3
                r2 = XM.mc_child_rng(r1, e2, L)
                push!(ids, XM.mc_stream_id(r2)); n += 1
                for e3 in 0:3
                    r3 = XM.mc_child_rng(r2, e3, L)
                    push!(ids, XM.mc_stream_id(r3)); n += 1
                    @test XM.mc_stream_layer(r3) == 3
                end
            end
        end
    end
    @test length(ids) == n == 5 * (1 + 4 + 16 + 64)
    # a child's stream does not depend on how far its parent's counter has advanced
    r = XM.mc_primary_rng(7, 3)
    _, radv = mc_uniform(r)
    @test XM.mc_child_rng(radv, 2, L) == XM.mc_child_rng(r, 2, L)
    # primaries keep the historical key: mc_primary_rng(seed, i) == MCRng(seed, i)
    @test XM.mc_primary_rng(0xBEEF, 12345) == MCRng(0xBEEF, 12345)
    @test mc_uniform(XM.mc_primary_rng(0xBEEF, 9))[1] == mc_uniform(MCRng(0xBEEF, 9))[1]
    # distinct identities in the counter words give distinct draws
    a = XM.mc_child_rng(XM.mc_primary_rng(1, 1), 0, XM.MCStreamLayout(10^6, 10^5, 3))
    @test mc_uniform(a)[1] != mc_uniform(XM.mc_primary_rng(1, 1))[1]
    # layouts that cannot hold the genealogy are refused
    @test_throws ArgumentError XM.MCStreamLayout(10^7, 100_000, 7)
    @test_throws ArgumentError XM.MCStreamLayout(10, 10, 16)
    # the production sphere layout at 10^7 primaries and three layers
    Lp = XM.MCStreamLayout(10^7, XM._MC_TRACK_MAX_STEPS, 3)
    @test Lp.gid_bits + 3 * Lp.event_bits <= 124
    # device split: every global id launched once, same stream set for 1..5 devices
    N = 1001
    streams(nd) = Set(XM.mc_stream_id(XM.mc_primary_rng(0x5ca77e5, g))
                      for s in _mc_device_shares(N, nd) for g in s.first:(s.first + s.n - 1))
    for nd in 1:5
        @test sum(s.n for s in _mc_device_shares(N, nd)) == N
        @test streams(nd) == streams(1)
        @test length(streams(nd)) == N
    end
end

@testset "thin sphere: device kernel on the KA CPU backend" begin
    sph = SphereModel()
    run(d) = mc_spectrum_gpu(sph, 20_000; devices = xe_hosts(d), seed = 0xC0FFEE, n_ebins = 60)
    g1 = run(1)
    c1 = mc_spectrum(sph, 20_000; nthreads = 1, seed = 0xC0FFEE, n_ebins = 60)
    @test sum(g1.count) > 1000
    @test g1.count == c1.count
    @test xe_bits(g1.sumw) == xe_bits(c1.sumw)       # device kernel == CPU driver, bitwise
    @test xe_bits(g1.sumwE) == xe_bits(c1.sumwE)
    for nd in (2, 5)
        g = run(nd)
        @test g.count == g1.count
        @test xe_bits(g.sumw) == xe_bits(g1.sumw)
        @test xe_bits(g.sumwE) == xe_bits(g1.sumwE)
    end
    for nt in (4, 16)
        c = mc_spectrum(sph, 20_000; nthreads = nt, seed = 0xC0FFEE, n_ebins = 60)
        @test xe_bits(c.sumwE) == xe_bits(c1.sumwE)
    end
    # workgroup size is a launch detail only
    gw = mc_spectrum_gpu(sph, 20_000; devices = xe_hosts(2), seed = 0xC0FFEE, n_ebins = 60,
                         workgroupsize = 7)
    @test xe_bits(gw.sumwE) == xe_bits(g1.sumwE)
end

@testset "scattering sphere: device kernels on the KA CPU backend" begin
    tbl = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))   # Thomson table
    ssph = ScatteringSphere(n_e = 5.0e18)             # tau_es ~ 1: every layer populated
    kw = (seed = 0xBEEF, n_ebins = 60, max_scatt = 3)
    run(d; o...) = _mc_scatter_gpu(ssph, tbl, 2_000; devices = xe_hosts(d),
                                   collect_children = true, kw..., o...)
    g1, k1 = run(1)
    @test g1.n_children > 2_000                       # several layers populated
    @test all(!isempty, k1)
    @test sum(length, k1) == g1.n_children
    # every child identity is unique
    @test length(Set(first.(reduce(vcat, k1)))) == g1.n_children
    b1 = xe_scatter_bits(g1)
    ga, ka = run(1)
    @test xe_scatter_bits(ga) == b1 && ka == k1       # run to run
    for nd in (2, 5)
        g, k = run(nd)
        @test xe_scatter_bits(g) == b1                # spectra + diagnostics
        @test xe_counts(g) == xe_counts(g1)
        @test k == k1                                 # identical children, all layers
    end
    # child buffers far too small: every layer overflows and is rerun, nothing is dropped
    go, ko = run(2; capacity_factor = 0.01, min_capacity = 1)
    @test xe_scatter_bits(go) == b1 && ko == k1
    # CPU driver: same photons, same children, same bits, for any chunk count
    for nt in (1, 4, 16)
        c = mc_spectrum_scatter(ssph, tbl, 2_000; nthreads = nt, kw...)
        @test xe_scatter_bits(c) == b1
        @test xe_counts(c) == xe_counts(g1)
    end
end

end
