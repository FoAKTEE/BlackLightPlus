# Oracle: BLP.MonteCarlo, the Monte Carlo builders of BLP.Data and the exact accumulators
# of BLP.Execution against BlackLightPlus.MonteCarlo and the old device code paths
# (`src/montecarlo/mc_gpu.jl`), bit for bit: emissivities, emission tetrad, hot cross
# section and its table, generation tables, generation (zone, sphere, old device
# generator), the Compton kick, the straight-line tracker with and without scattering
# (old CPU and device marches, children with their streams), the curved tracker on the
# torus over Kerr, and the exact spectra under threads of 1, 3 and 8 Julia threads and
# several chunkings and on pools of 1, 2, 3 and 5 host devices.
#
# The script runs twice. Natively the comparisons marked `contracted` agree to
# round-off: the torus zones, tracks and inclination binning, where the plasma decode
# and the metric products of the new code are inlined into different surroundings than
# the old `plasma_at`, so LLVM fuses a different set of the `muladd`s of the StaticArrays
# products into FMA instructions. The rerun on `-C generic` (no FMA) requires every
# comparison, the `contracted` ones included, to be bitwise.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "decks.jl"))

using BLP.MonteCarlo
using BLP.MonteCarlo: record!, spawn!
using BLP.Execution: ThreadedExecutor, PoolExecutor, ExactBins, PolarBands, InclinationBands,
                     Spectrum, nuLnu, nuLnu_total
import BLP.Execution
const NMa = BLP.Matter
const ND = BLP.Data
const OS = BlackLightPlus.Simulation
const DK = Decks
const bits = Decks.bits

"True in the child process on a CPU target without FMA."
const NOFMA = get(ENV, "BLP_ORACLE_NOFMA", "") == "1"
"Relative tolerance of a natively `contracted` comparison."
const CTOL = 1e-9

@noinline callf(f::F, args...) where {F} = f(args...)
@noinline callk(f::F, args...; kw...) where {F} = f(args...; kw...)
allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

# Field tuples of records of either package.
fields(a) = ntuple(k -> getfield(a, k), fieldcount(typeof(a)))
same_f(a, b) = fields(a) === fields(b)
near_f(a, b) = all(map(fields(a), fields(b)) do x, y
    x isa AbstractFloat || x isa StaticArray ? isapprox(x, y; rtol = CTOL) : x == y
end)
agree_f(a, b; contracted = false) = (NOFMA || !contracted) ? same_f(a, b) : near_f(a, b)
tonew(sp) = Superphoton(fields(sp)...)
tonew_rng(r) = BLP.Numerics.MCRng(r.key, r.ctr, r.sid)
tonew_zone(z) = NMa.EmissionZone{Float64}(fields(z)...)

# The old device code: the sphere generator, the straight-line marches and the binning.
const DEV = scratch_module(:OldMCDevice)
Core.eval(DEV, quote
    using BlackLightPlus: MCRng, mc_uniform, sample_direction, int_jnu_zone, j_nu, jmax, alpha_nu,
                          scatter_superphoton, bias_func
    using BlackLightPlus.Constants: ME, CL, HPL
    using LinearAlgebra: transpose
    const _MCM = BlackLightPlus.MonteCarlo
    const _XSLOT = _MCM.XSUM_NSLOT
end)
load_old!(DEV, "montecarlo/mc_gpu.jl", (:_mc_gen_int_jnu, :_mc_gen_sphere_photon, :_mc_nu_alpha_dev,
                                        :_mc_track_thin_w, :_xsum_atomic_add!, :_mc_record_dev!,
                                        :_mc_nu_alpha_scatt_dev, :_mc_scatter_mom,
                                        :_mc_march_scatter!))

const HOTCROSS_DAT = get(ENV, "BLP_REF_GPUMONTY",
                         "/data/haiyangw/claude/Mahakala/ref-code/gpumonty/table/hotcross.dat")

# ------------------------------------------------------------------ the torus in both packages

function old_amr(; nc = 6, half = 12.0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, nc, nc, nc, nmb, 8)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; xl = SVector{3,Float64}[]
    dxs = SVector{3,Float64}[]; lloc = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        push!(lo, l); push!(hi, l .+ w); push!(xl, l .+ dx / 2); push!(dxs, SVector(dx, dx, dx))
        lloc[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            p = DK.torus_primitives(xl[m][1] + (i - 1) * dx, xl[m][2] + (j - 1) * dx,
                                    xl[m][3] + (k - 1) * dx)
            for q in 1:8
                data[i, j, k, m, q] = p[q]
            end
        end
    end
    return OS.AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), lloc,
                              collect(String, BLP.Data.PRIM_NAMES), nc, nc, nc, nmb)
end

# The old GRMHD model on the torus with the plasma of `Decks.torus_params`.
function old_model()
    st = OG.KerrSchildCartesian(DK.TORUS_A)
    L = NN.GNEWT * DK.TORUS_MBH / NN.CL^2
    u = OS.units(DK.TORUS_MBH, DK.TORUS_DENS * L^3)
    cfg = OR.SimCoeffConfig(bh_a = DK.TORUS_A, d_unit = u.dens_unit, e_unit = u.dens_unit * NN.CL^2,
                            b_unit = u.B_unit, plasma_mu = 0.5, plasma_ne_ni = 1.0, plasma_use_p = 1,
                            plasma_rat_low = 1.0, plasma_rat_high = 10.0, plasma_gamma = 13.0 / 9.0,
                            plasma_gamma_i = 5.0 / 3.0, plasma_gamma_e = 4.0 / 3.0,
                            plasma_thermal_frac = 1.0, cut_sigma_max = 1.0, cut_theta_e_min = -1.0)
    return OM.GRMHDModel(st, old_amr(), cfg, 13.0 / 9.0, u.L_unit, OG.horizon_radius(st), 100.0)
end

# ------------------------------------------------------------------ sphere references

# The old thin-sphere spectrum of primaries 1:N as exact limbs.
function old_thin_spectrum(so, wgt, dnmax, N, seed; n_ebins = 60)
    xs = OM.ExactSpectrum(; n_ebins = n_ebins, n_thbins = 6)
    for i in 1:N
        sp, rng = OM.sample_sphere_photon(so, wgt, dnmax, OM.mc_primary_rng(seed, i))
        sp, rec = OM.track_superphoton(sp, so, rng)
        rec && OM.record!(xs, sp)
    end
    return xs
end

# The thin-sphere limbs computed in a child process with `nt` Julia threads.
function child_thin_limbs(nt, chunk, Ns, seed, n_ebins)
    out = tempname()
    code = """
        using BLP
        include($(repr(joinpath(@__DIR__, "..", "decks.jl"))))
        s = BLP.Matter.UniformSphere()
        tabs, N = Decks.sphere_tables(s, $Ns)
        b = Decks.sphere_thin_bins(BLP.Execution.ThreadedExecutor(chunk = $chunk), s, tabs, N,
                                   $seed; n_ebins = $n_ebins)
        open(io -> (write(io, b.count); write(io, b.sumw); write(io, b.sumwE)), $(repr(out)), "w")
        """
    target = NOFMA ? ["-C", "generic"] : String[]
    cmd = `$(Base.julia_cmd()) $target -t $nt --project=$(Base.active_project()) -e $code`
    run(pipeline(cmd; stdout, stderr))
    return read(out)
end

limb_bytes(b) = vcat(reinterpret(UInt8, vec(b.count)), reinterpret(UInt8, vec(b.sumw)),
                     reinterpret(UInt8, vec(b.sumwE)))

@testset "oracle: MonteCarlo$(NOFMA ? " (no FMA)" : "")" begin
    rng = Xoshiro(20261016)

    @testset "emissivities and emission tetrad" begin
        nbad = 0
        for _ in 1:2000
            ne = 10.0^(3 + 12rand(rng)); te = 10.0^(-1 + 3rand(rng)); B = 10.0^(-1 + 3rand(rng))
            nu = 10.0^(8 + 8rand(rng)); s = rand(rng)
            nbad += !(callf(j_nu, ne, te, B, nu, s) === callf(OM.j_nu, ne, te, B, nu, s))
            nbad += !(callf(alpha_nu, ne, te, B, nu, s) === callf(OM.alpha_nu, ne, te, B, nu, s))
            nbad += !(callf(jmax, ne, te, B, nu) === callf(OM.jmax, ne, te, B, nu))
        end
        for _ in 1:200
            ne = 10.0^(3 + 12rand(rng)); te = 10.0^(-1 + 3rand(rng)); B = 10.0^(-1 + 3rand(rng))
            nu = 10.0^(8 + 8rand(rng))
            nbad += !(callf(int_jnu_zone, ne, te, B, nu) === callf(OM.int_jnu_zone, ne, te, B, nu))
        end
        st = NS.KerrSchildCartesian(0.9)
        for k in 1:2000
            x = SVector(randn(rng), randpoint(rng, 2.5, 30.0)...)
            gcov, gcon = NS.metric_and_inverse(st, x)
            fv = NMa.primitive_four_vectors(SVector(0.3randn(rng), 0.3randn(rng), 0.3randn(rng)),
                                            SVector(randn(rng), randn(rng), randn(rng)), gcov, gcon)
            bcon = k % 50 == 0 ? zero(SVector{4,Float64}) : fv.bcon
            nbad += !(callf(emission_tetrad, fv.ucon, fv.ucov, bcon, gcov, gcon) ===
                      callf(OM.emission_tetrad, fv.ucon, fv.ucov, bcon, gcov, gcon))
        end
        @test nbad == 0
    end

    @testset "hot cross section" begin
        nbad = 0
        for _ in 1:5000
            e = 10.0^(-6 + 9rand(rng)); mu = 2rand(rng) - 1; g = 1 + 10.0^(-3 + 4rand(rng))
            th = 10.0^(-4 + 6rand(rng))
            nbad += !(callf(kn_total, e) === callf(OM.kn_total, e))
            nbad += !(callf(boostcross, e, mu, g) === callf(OM.boostcross, e, mu, g))
            nbad += !(callf(dNdgamma_e, th, g) === callf(OM.dNdgamma_e, th, g))
        end
        for _ in 1:100
            e = 10.0^(-13 + 29rand(rng)); th = 10.0^(-5 + 9rand(rng))
            nbad += !(callf(total_compton_cross_num, e, th) === callf(OM.total_compton_cross_num, e, th))
        end
        @test nbad == 0
        tn = ND.build_hotcross_table(); to = OM.build_hotcross_table()
        @test bits(tn.logsigma) == bits(to.logsigma)
        if isfile(HOTCROSS_DAT)
            gn = ND.load_hotcross(HOTCROSS_DAT); go = OM.load_hotcross(HOTCROSS_DAT)
            @test bits(gn.logsigma) == bits(go.logsigma)
            nbad = 0
            for _ in 1:20_000
                e = 10.0^(-14 + 31rand(rng)); th = 10.0^(-5 + 10rand(rng))
                nu = 10.0^(8 + 16rand(rng)); ne = 10.0^(3 + 12rand(rng))
                nbad += !(callf(hotcross_lookup, gn, e, th) === callf(OM.hotcross_lookup, go, e, th))
                nbad += !(callf(kappa_es, gn, nu, th) === callf(OM.kappa_es, go, nu, th))
                nbad += !(callf(alpha_inv_scatt, gn, nu, th, ne) === callf(OM.alpha_inv_scatt, go, nu, th, ne))
            end
            @test nbad == 0
        else
            @info "gpumonty hotcross.dat not found: file and lookup comparisons skipped" path = HOTCROSS_DAT
        end
    end

    model = old_model()
    blocks = DK.amr_deck().blocks
    stn = NS.KerrSchildCartesian(DK.TORUS_A)
    params = DK.torus_params()
    zo = OM.emission_zones(model)
    zn = DK.torus_zones(blocks, stn, params)
    zc = tonew_zone.(zo)                       # the old zones as new records

    @testset "torus zones and generation tables" begin
        @test length(zn) == length(zo) > 600
        @test all(agree_f(zn[i], zo[i]; contracted = true) for i in eachindex(zo))
        jo = OM.build_zone_emission_table(zo); jn = ND.build_zone_emission_table(zc)
        @test bits(jn.jnu) == bits(jo.jnu) && bits(jn.dV) == bits(jo.dV)
        @test same_f((jn.lnu_min, jn.lnu_max, jn.dlnu, jn.n_esamp), (jo.lnu_min, jo.lnu_max, jo.dlnu, jo.n_esamp))
        wo = OM.build_weight_table(jo, 1000.0; nthreads = 3); wn = ND.build_weight_table(jn, 1000.0; nthreads = 5)
        @test bits(wn.lwgt) == bits(wo.lwgt)
        ws_o = OM.build_weight_table(zo[1:40], 77.0); ws_n = ND.build_weight_table(zc[1:40], 77.0)
        @test bits(ws_n.lwgt) == bits(ws_o.lwgt) && (ws_n.lnu_min, ws_n.dlnu) === (ws_o.lnu_min, ws_o.dlnu)
        No, dno = OM.zone_counts(jo, wo); Nn, dnn = ND.zone_counts(jn, wn)
        @test bits(Nn) == bits(No) && bits(dnn) == bits(dno)
        @test all(callf(zone_nsuperphotons, zc[z], wn) === callf(OM.zone_nsuperphotons, zo[z], wo)
                  for z in 1:25:length(zo))
        go = OM.build_int_jnu_table(zo[50], wo); gn = ND.build_int_jnu_table(zc[50], wn)
        @test bits(gn.ljnu) == bits(go.ljnu) && (gn.lnu_min, gn.dlnu) === (go.lnu_min, go.dlnu)
        @test tabulated_dnmax(gn, wn) === OM.tabulated_dnmax(go, wo)
        lo_, do_ = OM.zone_int_jnu_tables(jo, wo); ln_, dn_ = ND.zone_int_jnu_tables(jn, wn)
        @test bits(ln_) == bits(lo_) && bits(dn_) == bits(do_)
    end

    # ------------------------------------------------------------------ generation
    so = OM.SphereModel(); sn = NMa.UniformSphere()
    swo = OM.build_weight_table([OM.sphere_zone(so, SVector(0.0, 0.0, 0.0, 0.0))], 1.0e4)
    swn = ND.build_weight_table([table_zone(sn)], 1.0e4)
    _, sdn = OM.zone_nsuperphotons(OM.sphere_zone(so, SVector(0.0, 0.0, 0.0, 0.0)), swo)
    stabs = GenerationTables(swn, sdn)

    @testset "generation: sphere (CPU and device) and zones" begin
        @test bits(swn.lwgt) == bits(swo.lwgt)
        @test zone_nsuperphotons(table_zone(sn), swn) === OM.zone_nsuperphotons(OM.sphere_zone(so, zeros(SVector{4})), swo)
        E = OM.emission_tetrad(so.ucon, so.ucov, so.bcon, so.gcov, so.gcon)
        p = (R = so.R, reject_max = OM._MC_REJECT_MAX_ITER, lnu_min = swo.lnu_min,
             lnu_max = swo.lnu_max, dlnu = swo.dlnu, n_nodes = length(swo.lwgt), dnmax = sdn,
             weight_floor = OM._WEIGHT_FLOOR, ne = so.n_e, te = so.theta_e, B = so.B, E = E,
             gcov = so.gcov)
        nbad = 0; ndev = 0
        for g in 1:10_000
            a, ra = callf(OM.sample_sphere_photon, so, swo, sdn, OM.mc_primary_rng(7, g))
            b, rb = callf(generate, sn, stabs, BLP.Numerics.primary_rng(7, g))
            nbad += !same_f(a, b) + !same_f(ra, rb)
            x, pc, w, rd = callf(DEV._mc_gen_sphere_photon, OM.mc_primary_rng(7, g), p, swo.lwgt, nothing)
            ndev += !((x, pc, w) === (a.x, a.p, a.w)) + !same_f(rd, ra)
        end
        @test nbad == 0
        @test ndev == 0                        # the old device generator is the CPU arithmetic
        wo = OM.build_weight_table(zo, 1000.0); wn = ND.build_weight_table(zc, 1000.0)
        tz = 1:53:length(zo)                   # zones of the tabulated draws
        gto = Dict(z => OM.build_int_jnu_table(zo[z], wo) for z in tz)
        gtn = Dict(z => ND.build_int_jnu_table(zc[z], wn) for z in tz)
        nbad = sum(bits(gtn[z].ljnu) != bits(gto[z].ljnu) for z in tz)
        for g in 1:10_000
            z = 1 + (g * 7919) % length(zo)
            _, dn = OM.zone_nsuperphotons(zo[z], wo)
            a, ra = callf(OM.sample_zone_photon, zo[z], wo, dn, OM.mc_primary_rng(5, g))
            b, rb = callf(generate, zc[z], GenerationTables(wn, dn), BLP.Numerics.primary_rng(5, g))
            nbad += !same_f(a, b) + !same_f(ra, rb)
            if g % 10 == 0                     # tabulated generation
                z = tz[1 + g % length(tz)]
                go = gto[z]; gn = gtn[z]
                dt = OM.tabulated_dnmax(go, wo)
                a, ra = callf(OM.sample_zone_photon, zo[z], wo, dt, OM.mc_primary_rng(5, g), go)
                b, rb = callf(generate, zc[z], GenerationTables(wn, dt, gn), BLP.Numerics.primary_rng(5, g))
                nbad += !same_f(a, b) + !same_f(ra, rb)
            end
        end
        @test nbad == 0
    end

    @testset "Compton kick" begin
        nbad = 0
        st = NS.KerrSchildCartesian(0.9)
        for k in 1:10_000
            te = 10.0^(-2 + 3rand(rng))
            k0 = 10.0^(-7 + 8rand(rng))
            kt = SVector(k0, (k0 * normalize(SVector(randn(rng), randn(rng), randn(rng))))...)
            ro = OM.MCRng(11, k); rn = BLP.Numerics.MCRng(11, k)
            a = callf(OM.compton_scatter, kt, te, ro); b = callf(compton_scatter, kt, te, rn)
            nbad += !(a[1] === b[1]) + !same_f(a[2], b[2])
            a = callf(OM.sample_electron, te, kt, ro); b = callf(sample_electron, te, kt, rn)
            nbad += !(a[1] === b[1]) + !same_f(a[2], b[2])
            a = callf(OM.sample_scattered_energy, k0, ro); b = callf(sample_scattered_energy, k0, rn)
            nbad += !(a[1:2] === b[1:2]) + !same_f(a[3], b[3])
            x = SVector(0.0, randpoint(rng, 3.0, 20.0)...)
            gcov, gcon = NS.metric_and_inverse(st, x)
            fv = NMa.primitive_four_vectors(SVector(0.3randn(rng), 0.3randn(rng), 0.3randn(rng)),
                                            SVector(randn(rng), randn(rng), randn(rng)), gcov, gcon)
            u = fv.ucon; b4 = fv.bcon
            pc, _ = NS.null_covariant(st, x, SVector(randn(rng, 3)...))
            a = callf(OM.scatter_superphoton, pc, u, gcov * u, b4, te, gcov, gcon, ro)
            b = callf(scatter_superphoton, pc, u, gcov * u, b4, te, gcov, gcon, rn)
            nbad += !(a[1:2] === b[1:2]) + !same_f(a[3], b[3])
        end
        @test nbad == 0
    end

    # ------------------------------------------------------------------ straight-line tracker
    ctl = FlatMarch(sn)
    stm = NS.Minkowski()

    @testset "sphere: absorbing march against the CPU and device trackers" begin
        nbad = 0; ndev = 0; nrec = 0
        for g in 1:5000
            a, _ = OM.sample_sphere_photon(so, swo, sdn, OM.mc_primary_rng(3, g))
            o, ro = callf(OM.track_superphoton, a, so, OM.MCRng(1, 1))
            n = callk(track!, ctl, stm, sn, Opacities(), tonew(a), BLP.Numerics.MCRng(1, 1), NullSink())
            nbad += !same_f(o, n.sph) + (ro != n.recorded)
            nrec += n.recorded
            wd = callf(DEV._mc_track_thin_w, a.x, a.p, a.w, so.gcon, so.n_e, so.theta_e, so.B,
                       so.R, OM._MC_STEP_FRAC, OM._MC_TRACK_MAX_STEPS)
            ndev += !(wd === n.sph.w)
        end
        @test nbad == 0 && ndev == 0 && nrec == 5000
    end

    if isfile(HOTCROSS_DAT)
        gn = ND.load_hotcross(HOTCROSS_DAT); go = OM.load_hotcross(HOTCROSS_DAT)
        @testset "sphere: scattering march, layer by layer, against the CPU and device marches" begin
            K = 3; N = 1500
            ssn = NMa.UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
            sso = OM.ScatteringSphere(n_e = 5.0e18)
            wz = OM.build_weight_table([OM.sphere_zone(sso, zeros(SVector{4}))], 1.0e3; nu_max = 1.0e13)
            _, dz = OM.zone_nsuperphotons(OM.sphere_zone(sso, zeros(SVector{4})), wz)
            lo = OM.MCStreamLayout(N, OM._MC_TRACK_MAX_STEPS, K)
            ln = BLP.Numerics.MCStreamLayout(N, SPHERE_MAX_STEPS, K)
            opac = Opacities(; table = gn, bias = Bias(ssn), max_scatt = K)
            cs = FlatMarch(ssn)
            pdev = (ne = sso.n_e, te = sso.theta_e, R = sso.R, gcon = sso.gcon, gcov = sso.gcov,
                    ucon = sso.ucon, ucov = sso.ucov, bcon = sso.bcon, step_frac = OM._MC_STEP_FRAC,
                    track_max = OM._MC_TRACK_MAX_STEPS, weight_min = OM._MC_WEIGHT_MIN,
                    max_scatt = K, layout = lo)
            layer = [OM.sample_sphere_photon(sso, wz, dz, OM.mc_primary_rng(9, g)) for g in 1:N]
            nbad = 0; ndev = 0; nchild = 0
            for depth in 0:K
                isempty(layer) && break
                next = Any[]
                esc_o = Set{UInt128}(); esc_n = Set{UInt128}()
                for (sp, r) in layer
                    ws = OM.ScatterWorkspace(lo)
                    e, rec, loss, ro = callk(OM.track_superphoton_scatter!, ws, sp, sso, go, r; max_scatt = K)
                    sink = VectorSink(ln)
                    n = callk(track!, cs, stm, ssn, opac, tonew(sp), tonew_rng(r), sink;
                              absorb = Val(false), scatter = Val(true))
                    nbad += !same_f(e, n.sph) + (rec != n.recorded) + !same_f(ro, n.rng)
                    nbad += !(sink.ledger[1][2] === loss) + (length(ws.sp) != length(sink.children))
                    rec && push!(esc_o, OM.mc_stream_id(r)); n.recorded && push!(esc_n, BLP.Numerics.stream_id(tonew_rng(r)))
                    for (k, (c, cr)) in enumerate(sink.children)
                        k <= length(ws.sp) || break
                        nbad += !same_f(ws.sp[k], c) + !same_f(ws.rngs[k], cr)
                    end
                    # the old device march: the same children (slot order), weight and ledger
                    buf = (x = Vector{SVector{4,Float64}}(undef, 4096), p = Vector{SVector{4,Float64}}(undef, 4096),
                           w = zeros(4096), n = zeros(Int32, 4096), r = Vector{OM.MCRng}(undef, 4096))
                    ctr = zeros(Int64, 1)
                    drec, dw, dloss, dchild = callf(DEV._mc_march_scatter!, sp.x, sp.p, sp.w, sp.nscatt, r,
                                                    pdev, go.logsigma, buf, ctr, 4096)
                    ndev += (drec != rec) + !(dw === e.w) + !(dloss === loss) + (ctr[1] != length(sink.children))
                    for (k, (c, cr)) in enumerate(sink.children)
                        k <= ctr[1] || break
                        ndev += !((buf.x[k], buf.p[k], buf.w[k], buf.n[k]) === (c.x, c.p, c.w, c.nscatt)) +
                                !same_f(buf.r[k], cr)
                    end
                    append!(next, zip(ws.sp, ws.rngs))
                end
                nbad += esc_o != esc_n
                nchild += length(next)
                layer = next
            end
            @test nbad == 0 && ndev == 0 && nchild > 1000
        end
    else
        @info "gpumonty hotcross.dat not found: scattering-march comparisons skipped" path = HOTCROSS_DAT
    end

    # ------------------------------------------------------------------ curved tracker
    @testset "torus: absorbing and scattering curved marches" begin
        wo = OM.build_weight_table(OM.build_zone_emission_table(zo), 2000 / log(1.0e8))
        _, dno = OM.zone_counts(OM.build_zone_emission_table(zo), wo)
        C = 32.0; R_rec = 1.0e3
        Rem = max(100.0, maximum(z -> 1.2 * OG.ks_radius(model.spacetime, z.x), zo))
        cc = CurvedMarch(stn; C = C, escape = R_rec, R_em = Rem)
        seeds = map(1:2000) do g
            z = 1 + (g * 7919) % length(zo)
            OM.sample_zone_photon(zo[z], wo, dno[z], OM.mc_primary_rng(5, g))
        end
        incl = InclinationBands((17.0, 50.0, 90.0), 10.0)
        bo = [OM.ExactSpectrum(; nu_min = 1e8, nu_max = 1e24, n_ebins = 120, n_thbins = 1) for _ in 1:3]
        to = OM.ExactSpectrum(; nu_min = 1e8, nu_max = 1e24, n_ebins = 120, n_thbins = 1)
        bn = ExactBins(incl; nu_min = 1e8, nu_max = 1e24, n_ebins = 120)
        lo_, hi_ = collect(incl.lo), collect(incl.hi)
        nbad = 0; nrec = 0
        for (sp, _) in seeds
            o, ro = callf(OM.track_superphoton_grmhd, sp, model, C, R_rec, Rem)
            n = callk(track!, cc, stn, blocks, Opacities(params), tonew(sp), BLP.Numerics.MCRng(1, 1), NullSink())
            nbad += !agree_f(o, n.sph; contracted = true) + (ro != n.recorded)
            nrec += ro
            if ro
                OM._grmhd_record!(bo, to, o, model.spacetime, lo_, hi_)
                Execution.record!(bn, o.E, o.w, escape_direction(cc, tonew(o)))
            end
        end
        @test nbad == 0 && 1000 < nrec < 2000
        so_ = Spectrum(bn)
        @test so_.count == vcat((b.count for b in bo)..., to.count)
        if NOFMA
            @test all(bn.sumw[:, k, :] == (k <= 3 ? bo[k] : to).sumw[:, 1, :] for k in 1:4)
            @test all(bn.sumwE[:, k, :] == (k <= 3 ? bo[k] : to).sumwE[:, 1, :] for k in 1:4)
        else
            @test all(isapprox(so_.sumwE[k, :], OM.Spectrum(k <= 3 ? bo[k] : to).sumwE[1, :]; rtol = CTOL)
                      for k in 1:4)
        end
        if isfile(HOTCROSS_DAT)
            gn = ND.load_hotcross(HOTCROSS_DAT); go = OM.load_hotcross(HOTCROSS_DAT)
            K = 3; bias0 = 1.0e5
            lo = OM.MCStreamLayout(2000, OM._GRMHD_MAX_STEPS, K)
            ln = BLP.Numerics.MCStreamLayout(2000, CURVED_MAX_STEPS, K)
            opac = Opacities(params; table = gn, bias = Bias(bias0), max_scatt = K)
            layer = collect(seeds)
            nbad = 0; nchild = 0
            for depth in 0:K
                isempty(layer) && break
                next = Any[]
                for (sp, r) in layer
                    ws = OM.ScatterWorkspace(lo)
                    e, rec, ro = callf(OM.track_superphoton_grmhd_scatter!, ws, sp, model, go, C, R_rec,
                                       Rem, bias0, K, r)
                    sink = VectorSink(ln)
                    n = callk(track!, cc, stn, blocks, opac, tonew(sp), tonew_rng(r), sink;
                              absorb = Val(true), scatter = Val(true))
                    nbad += !agree_f(e, n.sph; contracted = true) + (rec != n.recorded) + !same_f(ro, n.rng)
                    nbad += length(ws.sp) != length(sink.children)
                    for (k, (c, cr)) in enumerate(sink.children)
                        k <= length(ws.sp) || break
                        nbad += !agree_f(ws.sp[k], c; contracted = true) + !same_f(ws.rngs[k], cr)
                    end
                    append!(next, zip(ws.sp, ws.rngs))
                end
                nchild += length(next)
                layer = next
            end
            @test nbad == 0 && nchild > 300
        end
    end

    # ------------------------------------------------------------------ exact spectra
    @testset "exact spectra under every executor" begin
        Ns = 20_000; seed = 0xC0FFEE
        tabs, N = DK.sphere_tables(sn, Ns)
        wo = OM.build_weight_table([OM.sphere_zone(so, zeros(SVector{4}))],
                                   Float64(Ns) / (log(1.0e16) - log(1.0e8)))
        _, dno = OM.zone_nsuperphotons(OM.sphere_zone(so, zeros(SVector{4})), wo)
        @test N == max(1, round(Int, OM.zone_nsuperphotons(OM.sphere_zone(so, zeros(SVector{4})), wo)[1]))
        xs = old_thin_spectrum(so, wo, dno, N, seed)
        old = OM.Spectrum(xs)
        drv = OM.mc_spectrum(so, Ns; nthreads = 3, seed = seed, n_ebins = 60)
        @test bits(old.sumwE) == bits(drv.sumwE) && old.count == drv.count
        execs = (ThreadedExecutor(chunk = 1), ThreadedExecutor(chunk = 37), ThreadedExecutor(chunk = 4096),
                 (PoolExecutor(DK.hosts(nd); chunk = 1999) for nd in DK.NDEV)...,
                 PoolExecutor(DK.hosts(3); chunk = 128))
        for exec in execs
            b = DK.sphere_thin_bins(exec, sn, tabs, N, seed; n_ebins = 60)
            @test b.count == xs.count && b.sumw == xs.sumw && b.sumwE == xs.sumwE
            s = Spectrum(b)
            @test bits(nuLnu(s)[2]) == bits(OM.nuLnu(old)[2])
            @test bits(nuLnu_total(s)[2]) == bits(OM.nuLnu_total(old)[2]) && nuLnu_total(s)[3] == OM.nuLnu_total(old)[3]
        end
        ref = limb_bytes(DK.sphere_thin_bins(ThreadedExecutor(chunk = 64), sn, tabs, N, seed; n_ebins = 60))
        for (nt, chunk) in ((1, 4096), (3, 97))
            @test child_thin_limbs(nt, chunk, Ns, seed, 60) == ref
        end
    end

    @testset "hot path" begin
        sph, r = generate(sn, stabs, BLP.Numerics.primary_rng(1, 1))
        @test allocs(generate, sn, stabs, BLP.Numerics.primary_rng(1, 2)) == 0
        zt = GenerationTables(swn, zone_nsuperphotons(zc[100], swn)[2])
        @test allocs(generate, zc[100], zt, BLP.Numerics.primary_rng(1, 2)) == 0
        trk(c, st, p, o, s, r, k, a, b) = track!(c, st, p, o, s, r, k; absorb = a, scatter = b)
        bins = ExactBins(PolarBands(6))
        sink = DK.BinsSink(ctl, (bins,), nothing, nothing)
        @test allocs(trk, ctl, stm, sn, Opacities(), sph, r, sink, Val(true), Val(false)) == 0
        cc = CurvedMarch(stn; C = 32.0, R_em = 30.0)
        zs, zr = generate(zc[100], zt, BLP.Numerics.primary_rng(2, 2))
        @test allocs(trk, cc, stn, blocks, Opacities(params), zs, zr, NullSink(), Val(true), Val(false)) == 0
        @test allocs(Execution.record!, bins, 1.0e-6, 1.0, SVector(0.1, 0.2, 0.3)) == 0
        buf = Execution.ChildBuffer(Superphoton, 8, BLP.Numerics.MCStreamLayout(8, 8, 2))
        @test allocs(Execution.spawn!, buf, sph, r, 1) == 0
        for x in (ctl, cc, Opacities(params; table = HotcrossTable(zeros(NW + 1, NT + 1))), bins, buf, stabs)
            @test Decks.device_isbits(x)
        end
    end

    if !NOFMA
        @testset "every comparison bitwise without FMA" begin
            cmd = `$(Base.julia_cmd()) -C generic -t 8 --project=$(Base.active_project())
                   -e "using Test; include($(repr(@__FILE__)))"`
            @test success(pipeline(addenv(cmd, "BLP_ORACLE_NOFMA" => "1"); stdout, stderr))
        end
    end
end
