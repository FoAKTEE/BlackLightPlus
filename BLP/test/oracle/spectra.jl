# Oracle: the Monte Carlo spectrum product `mc_spectrum` against the BlackLightPlus drivers
# for the same `Ns` and seeds: `mc_spectrum` (thin sphere, exact and tabulated
# generation), `mc_spectrum_scatter` (the scattering sphere), both methods of
# `mc_spectrum_gpu` on pools of host devices (the old device files are evaluated into a
# scratch module), and `mc_spectrum_grmhd` on the torus block deck over Kerr, absorbing
# and scattering, with the emission zones they start from. The product runs under
# threads and on host-device pools of 1, 2, 3 and 5 devices.
#
# Spectra (counts, Σ w, Σ w E of every bin) and diagnostics are compared bit for bit. The
# torus comparisons are marked `contracted`: the plasma decode and the metric products of
# the new code are inlined into different surroundings than the old `plasma_at`, so LLVM
# fuses a different set of the StaticArrays `muladd`s into FMA instructions (see
# `montecarlo.jl`). Natively they must agree to `MARCH_RTOL` of each array's maximum with
# equal counts; the script then reruns itself on `-C generic`, a CPU target without FMA,
# where every comparison must be bitwise.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "decks.jl"))

using BLP.Plans: RunConfig, prepare
using BLP.Products: mc_spectrum, emission_zones
using BLP.Matter: UniformSphere
using BLP.Execution: ThreadedExecutor, PoolExecutor
const ND = BLP.Data
const bits = Decks.bits

"True in the child process on a CPU target without FMA."
const NOFMA = get(ENV, "BLP_ORACLE_NOFMA", "") == "1"
"Largest deviation, over an array's largest old magnitude, of a natively `contracted` comparison."
const MARCH_RTOL = 1e-7
const DEVIATIONS = Dict{String,Float64}()

const OLD = old_device_module(:OldMCDevice, ("gpu_image.jl", joinpath("montecarlo", "mc_gpu.jl")))
old_hosts(nd) = [OLD.HostDevice(k) for k in 1:nd]

const EXECS = (ThreadedExecutor(chunk = 4096), ThreadedExecutor(chunk = 37),
               (PoolExecutor(Decks.hosts(nd); chunk = c) for (nd, c) in zip(Decks.NDEV, (512, 997, 300, 4096)))...)
exec_name(e::ThreadedExecutor) = "threads chunk $(e.chunk)"
exec_name(e::PoolExecutor) = "$(length(e.pool.devices)) host devices chunk $(e.pool.chunk)"

# ---------------------------------------------------------------- comparisons

"Largest absolute difference over the largest old magnitude (0 for equal arrays)."
function deviation(a::AbstractArray, b::AbstractArray)
    d = maximum(abs.(Float64.(a) .- Float64.(b)); init = 0.0)
    return d == 0.0 ? 0.0 : d / maximum(abs, b; init = floatmin())
end

"""
Whether the spectrum `new` equals the old spectrum `old`: equal counts and bitwise `Σ w`,
`Σ w E`; for a `contracted` comparison in the native process, equal counts and both sums
within `MARCH_RTOL` (the deviation recorded under `label`).
"""
function agree(label, new, old; contracted = false)
    if NOFMA || !contracted
        ok = new.count == old.count && bits(new.sumw) == bits(old.sumw) &&
             bits(new.sumwE) == bits(old.sumwE)
        ok || (DEVIATIONS[label] = max(deviation(new.sumw, old.sumw), deviation(new.sumwE, old.sumwE),
                                       new.count == old.count ? 0.0 : Inf))
        return ok
    end
    d = max(deviation(new.sumw, old.sumw), deviation(new.sumwE, old.sumwE))
    DEVIATIONS[label] = new.count == old.count ? d : Inf
    return new.count == old.count && d <= MARCH_RTOL
end

"Bitwise equality of scalars; natively within `MARCH_RTOL` when `contracted`."
function agree_scalars(label, new::Tuple, old::Tuple; contracted = false)
    (NOFMA || !contracted) && return bits(new) == bits(old)
    d = maximum(abs(x - y) / max(abs(y), floatmin()) for (x, y) in zip(new, old))
    DEVIATIONS[label] = d
    return d <= MARCH_RTOL
end

"The old `ScatterResult` against the new: orders, total, counts and the weight ledger."
agree_scatter(label, n, o) =
    all(agree("$label order $k", n.orders[k], o.orders[k]) for k in eachindex(o.orders)) &&
    length(n.orders) == length(o.orders) && agree("$label total", n.total, o.total) &&
    (n.max_scatt, n.n_primary, n.n_children) == (o.max_scatt, o.n_primary, o.n_children) &&
    agree_scalars("$label ledger", (n.w_primary, n.w_children, n.w_first_order, n.scatter_loss),
                  (o.w_primary, o.w_children, o.w_first_order, o.scatter_loss))

"The old `GRMHDSpectrumResult` against the new (`contracted`)."
function agree_grmhd(label, n, o)
    ok = n.incl_deg == o.incl_deg && n.half_width_deg == o.half_width_deg &&
         bits(n.dOmega) == bits(o.dOmega) && n.scattering == o.scattering &&
         (n.n_zones, n.n_gen, n.n_recorded) == (o.n_zones, o.n_gen, o.n_recorded)
    ok &= all(agree("$label band $k", n.spectra[k], o.spectra[k]; contracted = true)
              for k in eachindex(o.spectra))
    ok &= agree("$label total", n.total, o.total; contracted = true)
    ok &= agree_scalars("$label powers", (n.emitted_power, n.escaped_power),
                        (o.emitted_power, o.escaped_power); contracted = true)
    return ok
end

zone_fields(z) = ntuple(k -> getfield(z, k), fieldcount(typeof(z)))
agree_zone(a, b) = (NOFMA ? bits(zone_fields(a)) == bits(zone_fields(b)) :
                    all(isapprox(x, y; rtol = 1e-9) for (x, y) in zip(zone_fields(a), zone_fields(b))))

hotcross_tables() = isfile(Decks.HOTCROSS_DAT) ?
    (ND.load_hotcross(Decks.HOTCROSS_DAT), OM.load_hotcross(Decks.HOTCROSS_DAT)) :
    (ND.build_hotcross_table(), OM.build_hotcross_table())

@testset "oracle: Monte Carlo spectra$(NOFMA ? " (no FMA)" : "")" begin
    tn, to = hotcross_tables()
    @test bits(tn.logsigma) == bits(to.logsigma)

    # ------------------------------------------------------------ thin sphere
    @testset "thin sphere: mc_spectrum and mc_spectrum_gpu" begin
        Ns = 20_000; seed = 0xC0FFEE
        p = prepare(UniformSphere())
        so = OM.SphereModel()
        for tab in (false, true)
            old = OM.mc_spectrum(so, Ns; nthreads = 3, seed = seed, n_ebins = 60, tabulated_gen = tab)
            for exec in EXECS
                new = mc_spectrum(p, Ns; seed = seed, n_ebins = 60, tabulated = tab, backend = exec)
                @test agree("thin sphere tabulated=$tab $(exec_name(exec))", new, old)
            end
            for nd in Decks.NDEV
                dev = OLD.mc_spectrum_gpu(so, Ns; devices = old_hosts(nd), chunk = 1999, seed = seed,
                                          n_ebins = 60, tabulated_gen = tab)
                @test agree("thin sphere tabulated=$tab mc_spectrum_gpu $nd host devices",
                            mc_spectrum(p, Ns; seed = seed, n_ebins = 60, tabulated = tab,
                                        backend = PoolExecutor(Decks.hosts(nd); chunk = 1999)), dev)
            end
        end
        wide = mc_spectrum(p, Ns; seed = 7, nu_min = 1.0e9, nu_max = 1.0e15, n_esamp = 200,
                           n_ebins = 90, bands = BLP.Execution.PolarBands(4), backend = :cpu)
        @test agree("thin sphere band 1e9-1e15", wide,
                    OM.mc_spectrum(so, Ns; nthreads = 5, seed = 7, nu_min = 1.0e9, nu_max = 1.0e15,
                                   n_esamp = 200, n_ebins = 90, n_thbins = 4))
    end

    # ------------------------------------------------------------ scattering sphere
    @testset "scattering sphere: mc_spectrum_scatter and mc_spectrum_gpu" begin
        Ns = 4_000; seed = 0xBEEF
        sn = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        so = OM.ScatteringSphere(n_e = 5.0e18)
        p = prepare(sn)
        kw = (scattering = true, table = tn, seed = seed, nu_max = 1.0e13, spec_nu_max = 1.0e17,
              n_ebins = 140)
        old = OM.mc_spectrum_scatter(so, to, Ns; nthreads = 4, seed = seed)
        @test old.n_children > 2_000
        for exec in EXECS
            @test agree_scatter("scattering sphere $(exec_name(exec))",
                                mc_spectrum(p, Ns; kw..., backend = exec), old)
        end
        @test agree_scatter("scattering sphere capacity 1",
                            mc_spectrum(p, Ns; kw..., capacity = 1, backend = :cpu), old)
        for nd in Decks.NDEV
            dev = OLD.mc_spectrum_gpu(so, to, Ns; devices = old_hosts(nd), chunk = 1500, seed = seed,
                                      min_capacity = 64)
            @test agree_scatter("scattering sphere mc_spectrum_gpu $nd host devices",
                                mc_spectrum(p, Ns; kw..., backend = PoolExecutor(Decks.hosts(nd); chunk = 1500)),
                                dev)
        end
        old2 = OM.mc_spectrum_scatter(so, to, Ns; nthreads = 2, seed = 3, max_scatt = 2,
                                      spec_nu_max = 1.0e16, n_ebins = 70, n_thbins = 3, tabulated_gen = true)
        @test agree_scatter("scattering sphere max_scatt 2, tabulated",
                            mc_spectrum(p, Ns; kw..., seed = 3, max_scatt = 2, spec_nu_max = 1.0e16,
                                        n_ebins = 70, bands = BLP.Execution.PolarBands(3),
                                        tabulated = true, backend = :host_pool), old2)
    end

    # ------------------------------------------------------------ torus
    @testset "torus: emission zones and mc_spectrum_grmhd" begin
        model = old_model()
        snap = Decks.amr_deck()
        L = NN.GNEWT * Decks.TORUS_MBH / NN.CL^2
        plan = prepare(RunConfig(m_bh_cgs = Decks.TORUS_MBH, mass_scale_cgs = Decks.TORUS_DENS * L^3,
                                 frequency_hz = 2.3e11);
                       spacetime = NS.KerrSchildCartesian(Decks.TORUS_A), matter = snap)
        for zk in ((), (stride = 2,), (r_emit_max = 9.0,), (max_zones = 100,))
            zn = emission_zones(plan; zk...); zo = OM.emission_zones(model; zk...)
            @test length(zn) == length(zo) > 50 && all(agree_zone(zn[i], zo[i]) for i in eachindex(zo))
        end
        Ns = 2_000
        cases = (("absorbing", (scattering = false, seed = 0x51EED5)),
                 ("scattering", (scattering = true, seed = 0xC0FFEE, bias0 = 1.0e5)),
                 ("absorbing, stride 2, tabulated", (scattering = false, seed = 9, stride = 2,
                                                     tabulated_gen = true)))
        for (name, c) in cases
            old = OM.mc_spectrum_grmhd(model, Ns; nthreads = 4, n_ebins = 120, table = to, c...)
            @test old.n_recorded > 500
            kw = (scattering = c.scattering, seed = c.seed, n_ebins = 120, table = tn,
                  spec_nu_max = 1.0e24, bias = get(c, :bias0, 1.0), stride = get(c, :stride, 1),
                  tabulated = get(c, :tabulated_gen, false))
            for exec in (name == "absorbing, stride 2, tabulated" ? EXECS[1:1] : EXECS)
                @test agree_grmhd("torus $name $(exec_name(exec))",
                                  mc_spectrum(plan, Ns; kw..., backend = exec), old)
            end
        end
    end

    @info "Spectra oracle$(NOFMA ? " (no FMA)" : ""): deviations / array maximum (0 = bitwise)\n" *
          join(("$k: $v" for (k, v) in sort(collect(DEVIATIONS))), "\n")

    if !NOFMA
        @testset "every comparison bitwise without FMA" begin
            cmd = `$(Base.julia_cmd()) -C generic -t 8 --project=$(Base.active_project())
                   -e "using Test; include($(repr(@__FILE__)))"`
            @test success(pipeline(addenv(cmd, "BLP_ORACLE_NOFMA" => "1"); stdout, stderr))
        end
    end
end
