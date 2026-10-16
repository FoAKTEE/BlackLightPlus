# Oracle: the analytic sources, the g-disk intensity and the Monte Carlo sphere zone of
# BLP.Matter against BlackLightPlus.Sources/Radiation/MonteCarlo, and the hot-path
# contract of every provider (no allocation, isbits, isbits after adaptation).
#
# Bit for bit: the source records, each kinematic helper compiled on its own, the torus
# potential, the doughnut and Page–Thorne plasma scalars, the Page–Thorne flux and
# temperature, the hot-spot centre, the validity of every sample (DomainError points
# included), the vacuum states, the g-disk intensity on band landings, the sphere zone.
#
# To a tolerance: the four-vectors of the analytic states, and the hot-spot scalars,
# which go through a metric contraction. StaticArrays evaluates a matrix-vector
# product as a `muladd` chain, which LLVM contracts to fma or not according to how the
# SLP vectorizer packs the enclosing function. `source_plasma_at` evaluates the metric
# in its own body and `plasma_state` receives it, so the packing, and the last bits,
# differ. The control below recompiles the old body with `metric_and_inverse` in place
# of `metric` and `inverse_metric`: it deviates from the old function in the same way.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "decks.jl"))

const NMa = BLP.Matter
const OS = BlackLightPlus.Sources

# Each call is compiled as its own non-inlined specialization.
@noinline callf(f::F, args...) where {F} = f(args...)
# The result, or the exception type.
tryf(f, args...) = try callf(f, args...) catch err; typeof(err) end

allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

"Largest deviation of the four-vectors, relative to each vector's largest component."
const VTOL = 1e-12
"Largest deviation of the scalars, relative to the source's peak value."
const STOL = 1e-14

vdev(u, v) = (m = maximum(abs, v); m == 0 ? maximum(abs, u) : maximum(abs.(u .- v)) / m)

# CKS position of Boyer–Lindquist (r, θ, φ) at time t.
cks(a, t, r, θ, φ) = SVector(t, sin(θ) * (r * cos(φ) - a * sin(φ)),
                             sin(θ) * (r * sin(φ) + a * cos(φ)), r * cos(θ))

# Points of every source kind: a band about the equator in [rlo, rhi], the whole ball
# of radius 1.5 rhi, and the spin axis.
function source_points(rng, a, rlo, rhi, n; band = 0.6)
    xs = SVector{4,Float64}[]
    for k in 1:n
        t = 20 * randn(rng)
        if k % 10 == 0
            push!(xs, SVector(t, 0.0, 0.0, (2rand(rng) - 1) * rhi))
        elseif k % 4 == 0
            push!(xs, SVector(t, randpoint(rng, 0.0, 1.5rhi)...))
        else
            r = rlo + (rhi - rlo) * rand(rng)
            θ = π / 2 + band * (2rand(rng) - 1)
            push!(xs, cks(a, t, r, θ, 2π * rand(rng)))
        end
    end
    return xs
end

# Points about the hot-spot centre at random times, a fifth of them anywhere within
# twice the orbit, and the spin axis.
function spot_points(rng, src, n)
    xs = SVector{4,Float64}[]
    for k in 1:n
        t = 30 * randn(rng)
        c = OS.hotspot_center(src, t)
        if k % 5 == 0
            push!(xs, SVector(t, randpoint(rng, 0.0, 2 * src.r_orbit)...))
        else
            d = randpoint(rng, 0.0, 6 * src.R_spot)
            push!(xs, SVector(t, c[1] + d[1], c[2] + d[2], c[3] + d[3]))
        end
    end
    append!(xs, [SVector(t, 0.0, 0.0, z) for t in (0.0, 3.0) for z in (4.0, 8.0, 20.0)])
    return xs
end

fields_equal(n, o) = all(same(getfield(n, f), getfield(o, f)) for f in fieldnames(typeof(o)))

"""
States of the new and the old source at `xs`: mismatches of the spacetime pair, of the
validity or of a vacuum state (`nbad`), samples inside, states bitwise equal, samples
with bitwise scalars, thrown samples, and the largest scalar (relative to `scale`) and
four-vector deviations.
"""
function compare_states(srcn, srco, a, xs, scale)
    stn = NS.KerrSchildCartesian(a); sto = OG.KerrSchildCartesian(a)
    nbad = 0; nin = 0; nexact = 0; nscalar = 0; nthrow = 0; dsca = 0.0; dvec = 0.0
    for x in xs
        g = callf(NS.metric_and_inverse, stn, x)
        if !(same(g, (callf(OG.metric, sto, x), callf(OG.inverse_metric, sto, x))) &&
             same(callf(NS.ks_radius, stn, x), callf(OG.ks_radius, sto, x)))
            nbad += 1
            continue
        end
        n = tryf(NMa.plasma_state, srcn, stn, x, g, nothing)
        o = tryf(OS.source_plasma_at, srco, sto, x)
        if n isa Type || o isa Type
            nbad += !(n === o); nthrow += 1
            continue
        end
        n.valid == o.inside || (nbad += 1; continue)
        o.inside || (nbad += !(n === NMa.vacuum(Float64)); continue)
        nin += 1
        sn = (n.n_e, n.theta_e, n.B); so = (o.n_e, o.theta_e, o.B)
        nscalar += same(sn, so)
        nexact += same((sn..., n.ucon, n.ucov, n.bcon), (so..., o.ucon, o.ucov, o.bcon))
        dsca = max(dsca, maximum(abs.(sn .- so) ./ scale))
        dvec = max(dvec, vdev(n.ucon, o.ucon), vdev(n.ucov, o.ucov), vdev(n.bcon, o.bcon))
        # the record's own lowered field and norm
        dvec = max(dvec, vdev(n.bcov, g[1] * n.bcon), abs(n.b_sq - 1))
        nbad += !(n.sigma === 0.0 && n.beta_inv === 0.0)
    end
    return (; nbad, nin, nexact, nscalar, nthrow, dsca, dvec)
end

# Control: the old torus body with the metric pair of `metric_and_inverse`.
@noinline function old_torus_with_pair(src, st, x)
    gcov, gcon = OG.metric_and_inverse(st, x)
    r = OG.ks_radius(st, x)
    r < src.r_cusp && return OS._vacuum(gcov, gcon)
    Xtt, Xtp, Xpp = OS._tphi_scalars(gcov, x)
    W, Omega, den = OS._constant_l_potential(Xtt, Xtp, Xpp, src.l0)
    den < 0 || return OS._vacuum(gcov, gcon)
    w = (src.W_surface - W) / (src.W_surface - src.W_center)
    w > 0 || return OS._vacuum(gcov, gcon)
    w = min(w, one(w))
    n_e = src.n_e_center * w^src.polytrope_n
    theta_e = src.theta_e_center * w
    B = OS._bfield_from_beta(OS._gas_pressure(n_e, theta_e), src.beta)
    ucon, ucov, _ = OS._circular_velocity(gcov, x, Omega)
    bcon = OS._toroidal_direction(gcov, x, src.l0, ucon, ucov)
    return OS._plasma(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon)
end

const NPTS = 2000

const DOUGHNUTS = [(0.0, (r_center = 9.0, n_e_center = 1e7, theta_e_center = 40.0, beta = 8.0)),
                   (0.5, (lambda = 0.5,)),
                   (0.5, (r_center = 25.0,)),
                   (0.9375, (l0 = 3.0, polytrope_n = 3.0, beta = 3.0)),
                   (0.9, (lambda = 0.5, prograde = false)),
                   (0.9375, (lambda = 0.3, fill_factor = 0.5))]

const DISKS = [(0.0, (r_out = 40.0,)), (0.5, (r_out = 30.0, scale_height = 0.1)),
               (0.94, (mdot = 2.0, n_e_max = 3e5, theta_e_max = 4.0, beta = 2.0)),
               (0.9, (prograde = false, r_out = 60.0)), (-0.9, (r_out = 60.0,))]

const SPOTS = [(0.0, (r_orbit = 8.0, phi0 = 0.3, R_spot = 0.6, n_e0 = 1e6, theta_e0 = 50.0, B0 = 15.0)),
               (0.5, (r_orbit = 6.0, t0 = 4.0, R_spot = 1.2)),
               (0.9375, (r_orbit = 5.0, prograde = false, R_spot = 0.4)),
               (0.5, (ambient_n_e = 1e3, ambient_theta_e = 5.0, ambient_B = 1.0)),
               (0.9375, (r_orbit = 7.0, ambient_n_e = 2e4, ambient_theta_e = 1.0, ambient_B = 3.0))]

const DISK_KNOBS = [NamedTuple(),
                    (sub_kep = 0.8, betar = 0.5, betaphi = 0.7),
                    (gamma_p = 0.0, mu_p = 2.5, sigma_p = 1.0, redshift_power = 4.0, rin = 3.0)]

@testset "oracle: Matter" begin
    rng = Xoshiro(20261016)

    @testset "kinematic helpers" begin
        nbad = 0
        for _ in 1:2000
            a = 0.999 * rand(rng); st = NS.KerrSchildCartesian(a)
            x = cks(a, randn(rng), 2.0 + 30 * rand(rng), π * rand(rng), 2π * rand(rng))
            gcov = NS.metric(st, x)
            X = callf(NMa.tphi_scalars, gcov, x)
            nbad += !same(Tuple(X), callf(OS._tphi_scalars, gcov, x))
            l0 = 4 * randn(rng)
            p = callf(NMa.constant_l_potential, X..., l0)
            nbad += !same(Tuple(p), callf(OS._constant_l_potential, X..., l0))
            Ω = NS.keplerian_omega(a, NS.ks_radius(st, x))
            u = tryf(NMa.circular_velocity, gcov, x, Ω)
            uo = tryf(OS._circular_velocity, gcov, x, Ω)
            nbad += !(u isa Type ? u === uo : same(Tuple(u), uo))
            u isa Type && continue
            nbad += !same(callf(NMa.toroidal_direction, gcov, x, l0, u.ucon, u.ucov),
                          callf(OS._toroidal_direction, gcov, x, l0, u.ucon, u.ucov))
            ne = 1e6 * rand(rng); te = 50 * rand(rng)
            nbad += !same(callf(NMa.gas_pressure, ne, te), callf(OS._gas_pressure, ne, te))
            nbad += !same(callf(NMa.bfield_from_beta, ne, te), callf(OS._bfield_from_beta, ne, te))
            nbad += !same(callf(NMa.equatorial_point, a, x[2]), callf(OS._equatorial_point, a, x[2]))
        end
        f(r) = r^2 - 2.0
        nbad += !same(NMa.bisect(f, 0.0, 3.0), OS._bisect(f, 0.0, 3.0))
        nbad += !same(NMa.SIGMA_SB, OS._SIGMA_SB)
        @test nbad == 0
    end

    @testset "Polish doughnut" begin
        ctrl_exact = 0; ctrl_dev = 0.0; ctrl_n = 0
        for (a, kw) in DOUGHNUTS
            srcn = NMa.PolishDoughnut(a; kw...); srco = OS.PolishDoughnut(a; kw...)
            @test fields_equal(srcn, srco)
            xs = source_points(rng, srcn.a, 0.5 * srcn.r_cusp, srcn.r_out + 3.0, NPTS)
            c = compare_states(srcn, srco, srcn.a, xs, (srcn.n_e_center, srcn.theta_e_center, 1.0))
            @test c.nbad == 0
            @test NPTS ÷ 20 < c.nin < NPTS - NPTS ÷ 10
            @test c.nscalar == c.nin
            @test c.dvec <= VTOL
            @info "doughnut a = $a" c
            stn = NS.KerrSchildCartesian(srcn.a); sto = OG.KerrSchildCartesian(srcn.a)
            @test all(same(callf(NMa.torus_potential, srcn, stn, x), callf(OS.torus_potential, srco, sto, x))
                      for x in xs)
            for x in xs
                o = callf(OS.source_plasma_at, srco, sto, x)
                o.inside || continue
                p = old_torus_with_pair(srco, sto, x)
                ctrl_n += 1
                ctrl_exact += same((p.ucon, p.ucov, p.bcon), (o.ucon, o.ucov, o.bcon))
                ctrl_dev = max(ctrl_dev, vdev(p.ucon, o.ucon), vdev(p.ucov, o.ucov), vdev(p.bcon, o.bcon))
            end
        end
        @info "control: old torus body with metric_and_inverse" ctrl_n ctrl_exact ctrl_dev
        @test ctrl_dev <= VTOL
        l_ms = NS.keplerian_specific_angular_momentum(0.5, NS.isco_radius(0.5))
        @test_throws ArgumentError NMa.PolishDoughnut(0.5; l0 = 0.8 * l_ms)
        @test_throws ArgumentError OS.PolishDoughnut(0.5; l0 = 0.8 * l_ms)
    end

    @testset "Page–Thorne disk" begin
        for (a, kw) in DISKS
            srcn = NMa.PageThorneDisk(a; kw...); srco = OS.PageThorneDisk(a; kw...)
            @test fields_equal(srcn, srco)
            xs = source_points(rng, srcn.a, srcn.r_isco - 1.0, srcn.r_out + 3.0, NPTS;
                               band = 5 * srcn.scale_height)
            c = compare_states(srcn, srco, srcn.a, xs, (srcn.n_e_max, srcn.theta_e_max, 1.0))
            @test c.nbad == 0
            @test NPTS ÷ 10 < c.nin < NPTS - NPTS ÷ 10
            @test c.nscalar == c.nin
            @test c.dvec <= VTOL
            @info "disk a = $a" c
            rs = srcn.r_isco .+ 200 .* rand(rng, 500) .- 0.5
            @test all(same(NMa.page_thorne_flux(srcn, r), OS.page_thorne_flux(srco, r)) &&
                      same(NMa.page_thorne_temperature(srcn, r), OS.page_thorne_temperature(srco, r)) &&
                      same(NMa.page_thorne_temperature(srcn, r; flux_scale = 3.0e7),
                           OS.page_thorne_temperature(srco, r; flux_scale = 3.0e7)) for r in rs)
        end
    end

    @testset "equatorial hot spot" begin
        for (a, kw) in SPOTS
            srcn = NMa.EquatorialHotspot(a; kw...); srco = OS.EquatorialHotspot(a; kw...)
            @test fields_equal(srcn, srco)
            xs = spot_points(rng, srco, NPTS)
            c = compare_states(srcn, srco, a, xs, (srcn.n_e0, srcn.theta_e0, srcn.B0))
            @test c.nbad == 0
            @test c.nin > NPTS ÷ 4
            srcn.ambient_n_e > 0 || @test c.nin < NPTS - NPTS ÷ 10
            @test c.dvec <= VTOL && c.dsca <= STOL
            @info "hot spot a = $a" kw c
            ts = 50 .* randn(rng, 200)
            @test all(same(Tuple(NMa.hotspot_center(srcn, t)), OS.hotspot_center(srco, t)) for t in ts)
        end
    end

    @testset "partials" begin
        dmax = 0.0
        st = NS.KerrSchildCartesian(0.0); sto = OG.KerrSchildCartesian(0.0)
        for (xe, ro) in ((SVector(2.0, 8.0 - 0.4, 0.5, 0.0), 8.0), (SVector(-1.0, 5.5, 1.0, 0.3), 6.0))
            fn(r) = NMa.plasma_state(NMa.EquatorialHotspot(0.0; r_orbit = r, R_spot = 0.6), st, xe,
                                     NS.metric_and_inverse(st, xe), nothing).n_e
            fo(r) = OS.source_plasma_at(OS.EquatorialHotspot(0.0; r_orbit = r, R_spot = 0.6), sto, xe).n_e
            dn = ForwardDiff.derivative(fn, ro); do_ = ForwardDiff.derivative(fo, ro)
            dmax = max(dmax, abs(dn - do_) / abs(do_))
        end
        for (a, kw) in DOUGHNUTS[1:3]
            srcn = NMa.PolishDoughnut(a; kw...); srco = OS.PolishDoughnut(a; kw...)
            stn = NS.KerrSchildCartesian(srcn.a); sto = OG.KerrSchildCartesian(srcn.a)
            for r in (srcn.r_center, 0.5 * (srcn.r_center + srcn.r_out))
                x0 = cks(srcn.a, 0.0, r, π / 2 - 0.05, 0.7)
                gn(x) = (s = NMa.plasma_state(srcn, stn, x, NS.metric_and_inverse(stn, x), nothing);
                         SVector(s.n_e, s.theta_e, s.B, s.ucon[1], s.bcon[3]))
                go(x) = (s = OS.source_plasma_at(srco, sto, x); SVector(s.n_e, s.theta_e, s.B, s.ucon[1], s.bcon[3]))
                Jn = ForwardDiff.jacobian(gn, x0); Jo = ForwardDiff.jacobian(go, x0)
                dmax = max(dmax, maximum(abs.(Jn .- Jo) ./ (abs.(Jo) .+ maximum(abs.(Jo); dims = 2) .* 1e-6)))
            end
        end
        @info "partials: largest relative deviation" dmax
        @test dmax <= 1e-10
    end

    @testset "g-disk intensity" begin
        nbad = 0; nvalid = 0; ntot = 0
        grid = range(-10.0, 10.0; length = 24)
        for a in (-0.5, 0.0, 0.3, 0.94, 0.998), inc in (17.0, 60.0, 120.0), knobs in DISK_KNOBS
            diskn = NMa.EquatorialDisk(a; knobs...); disko = OR.EquatorialDisk(a; knobs...)
            srcfun = (r, νr, aa, λ, η) -> OR.equatorial_disk_intensity(disko, r, νr, aa, λ, η)
            for α in grid, β in grid
                prep = NS.pixel_prep(α, β, deg2rad(inc), a)
                for n in 0:2
                    l = NS.band_landing(prep, n)
                    iv = callf(NMa.intensity, diskn, l)
                    io = l.valid ? callf(OR.equatorial_disk_intensity, disko, l.r, l.νr, a, l.λ, l.η) : 0.0
                    ic = l.valid ? OG.equatorial_intensity(l.r, l.νr, a; source = srcfun, λ = l.λ, η = l.η) :
                         zero(l.r)
                    nbad += !(iv === io && iv === ic)
                    nvalid += l.valid && iv > 0; ntot += 1
                end
            end
        end
        @test nbad == 0
        @test nvalid > ntot ÷ 4

        nbad = 0
        for _ in 1:3000
            a = 0.999 * rand(rng); r = 1.0 + 40 * rand(rng); b = rand(rng, (-1, 1))
            λ = 10 * randn(rng); η = 30 * rand(rng)
            kw = (sub_kep = 0.5 + rand(rng), betar = rand(rng), betaphi = rand(rng))
            for (fnew, fold) in ((NMa.gfactor_disk, OR.gfactor_disk), (NMa.gfactor_circular, OR.gfactor_circular),
                                 (NMa.gfactor_plunge, OR.gfactor_plunge))
                vn = try fnew(r, a, b, λ, η; kw...) catch err; typeof(err) end
                vo = try fold(r, a, b, λ, η; kw...) catch err; typeof(err) end
                nbad += !(vn === vo)
            end
            nbad += !same(NMa.johnson_su_profile(r; mu = a, sigma = kw.sub_kep, gamma = -λ),
                          OR.johnson_su_profile(r; mu = a, sigma = kw.sub_kep, gamma = -λ))
            vn = try NMa.equatorial_disk_intensity(r, b, a, λ, η; kw...) catch err; typeof(err) end
            vo = try OR.equatorial_disk_intensity(r, b, a, λ, η; kw...) catch err; typeof(err) end
            nbad += !(vn === vo)
        end
        f_n(sk) = NMa.equatorial_disk_intensity(4.0, 1, 0.94, 1.2, 22.8; sub_kep = sk)
        f_o(sk) = OR.equatorial_disk_intensity(4.0, 1, 0.94, 1.2, 22.8; sub_kep = sk)
        nbad += !(ForwardDiff.derivative(f_n, 1.0) === ForwardDiff.derivative(f_o, 1.0))
        @test nbad == 0
    end

    @testset "uniform sphere zone" begin
        nbad = 0; nin = 0
        st = NS.Minkowski()
        for kw in (NamedTuple(), (n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5),
                   (n_e = 2.5e15, theta_e = 0.7, B = 12.0, R = 7.0))
            sn = NMa.UniformSphere(; kw...); so = OM.SphereModel(; kw...)
            nbad += !same((sn.n_e, sn.theta_e, sn.B, sn.R, sn.dV), (so.n_e, so.theta_e, so.B, so.R, so.dV))
            for _ in 1:NPTS
                x = SVector(randn(rng), randpoint(rng, 0.0, 1.6 * so.R)...)
                g = callf(NS.metric_and_inverse, st, x)
                nbad += !same(g, (so.gcov, so.gcon))
                nbad += !fields_equal(callf(NMa.emission_zone, sn, x, g), callf(OM.sphere_zone, so, x))
                s = callf(NMa.plasma_state, sn, st, x, g, nothing)
                inside = OM._inside_sphere(so, x)
                nin += inside
                nbad += s.valid != inside
                nbad += inside ? !same((s.n_e, s.theta_e, s.B, s.ucon, s.ucov, s.bcon),
                                       (so.n_e, so.theta_e, so.B, so.ucon, so.ucov, so.bcon)) :
                        !(s === NMa.vacuum(Float64))
            end
        end
        @test nbad == 0
        @test NPTS < nin < 2NPTS
    end

    @testset "hot path" begin
        a = 0.9375
        st = NS.KerrSchildCartesian(a)
        provs = (NMa.PolishDoughnut(a; lambda = 0.5), NMa.PageThorneDisk(a),
                 NMa.EquatorialHotspot(a), NMa.EquatorialHotspot(a; ambient_n_e = 1.0),
                 NMa.UniformSphere(; R = 30.0))
        xs = (SVector(0.0, 8.0, 1.0, 0.2), SVector(0.0, 0.0, 0.0, 9.0), SVector(0.0, 30.0, -20.0, 1.0))
        for p in provs, x in xs
            g = NS.metric_and_inverse(st, x)
            @test allocs(NMa.plasma_state, p, st, x, g, nothing) == 0
            @test allocs(NMa.sample, p, st, x, nothing) == 0
        end
        disk = NMa.EquatorialDisk(a)
        l = NS.band_landing(NS.pixel_prep(-4.0, 3.0, deg2rad(17.0), a), 1)
        @test allocs(NMa.intensity, disk, l) == 0
        s = NMa.UniformSphere()
        g = NS.metric_and_inverse(NS.Minkowski(), SVector(0.0, 0.1, 0.2, 0.3))
        @test allocs(NMa.emission_zone, s, SVector(0.0, 0.1, 0.2, 0.3), g) == 0
        for p in (provs..., disk, NMa.emission_zone(s, SVector(0.0, 0.1, 0.2, 0.3), g))
            @test isbits(p)
            @test Decks.device_isbits(p)
        end
    end
end
