# Polarized surface materials: the power-law synchrotron surface against Krang (its
# polarized material with the Stokes I of its intensity material and the far-hemisphere
# polar sign of `emission_coordinates`) and the AART polarimetric disk against AART's
# Walker–Penrose constant (gauge f_t = 1) and EVPA vector, at stored screen points; the
# intensity and polarized intensity of a single image; the disk's Stokes I equal to the
# g-disk's; vacuum at invalid landings; the hot-path contract.
using Test, BlackLightPlus, StaticArrays, ForwardDiff
using BlackLightPlus.Spacetime: surface_prep, surface_landing, image_crossings, Landing,
                                kerr_bl_metric, rest_components, zamo_components, zamo_vector,
                                penrose_walker, screen_polarization
using BlackLightPlus.Matter: SynchrotronSurface, PolarizedDisk, EquatorialDisk, intensity,
                             polarized_emission, disk_velocity, rest_photon, surface_angle

const Decks = Main.Decks
allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

# Krang's README model: field and flow of the cone θs (mirrored field on the far cone)
function readme_surface(a, θs)
    ι = 0.58; η = θs > π / 2 ? π - 2.64 : 2.64
    return SynchrotronSurface(a; theta_s = θs, b = (sin(ι) * cos(η), sin(ι) * sin(η), cos(ι)),
                              beta = 0.87, theta_v = π / 2, phi_v = -1.7, sigma = 0.73, R = 4.0,
                              p1 = 1.0, p2 = 4.0)
end

function order_stokes(m, α, β, θo, a, n)
    q = surface_prep(α, β, θo, a; ro = Inf)
    θs = surface_angle(m)
    m1, m2, k = image_crossings(q, θs, n)
    S = (0.0, 0.0, 0.0)
    for (j, mm) in enumerate((m1, m2))
        j > k && break
        l = surface_landing(q, θs, mm)
        S = S .+ polarized_emission(m, l, α, β, θo)
    end
    return S
end

@testset "polarized surfaces" begin
    @testset "synchrotron surface == Krang" begin
        # (a, θo°, θs°, α, β, n, I, Q, U) from Krang, results/blp/analytic_upgrade/closed_form
        ref = ((0.94, 17.0, 75.0, 5.0, 2.0, 0, 0.028763387403013624, 0.02359070067781939, 0.016456345172163062),
               (0.94, 17.0, 105.0, -4.0, 3.0, 0, 0.008733490981931201, -0.0041818323710634655, 0.007667212189042334),
               (0.9, 60.0, 150.0, 4.4, -6.4, 0, 8.9415188504972, 8.064384992506673, -1.5501085777412238),
               (0.9, 60.0, 30.0, 2.0, 4.0, 0, 0.20713135506983177, 0.12020507079696366, 0.12046575021555456),
               (0.99, 80.0, 90.0, 7.067669172932331, -1.2330827067669172, 1, 5.331173985442337, 0.3864518979016965, -5.317148765425443),
               (0.94, 17.0, 25.0, -0.8120300751879699, -0.17042606516290726, 0, 0.07237571009050488, 0.04976782887994384, 0.05254908771502544))
        for (a, inc, θsd, α, β, n, I, Q, U) in ref
            S = order_stokes(readme_surface(a, deg2rad(θsd)), α, β, deg2rad(inc), a, n)
            @test S[1] ≈ I rtol = 1e-11
            @test S[2] ≈ Q rtol = 1e-10
            @test S[3] ≈ U rtol = 1e-10
        end
    end

    @testset "polarized g-disk == AART's Walker–Penrose constant" begin
        # (a, θo°, α, β, Re κ, Im κ, EVPA_x, EVPA_y) from AART polarizationk.kappa, D = 1e15
        ref = ((0.94, 17.0, 3.6390977443609014, 1.8345864661654119, 2.3632145306690022, 10.118502505104097, 1.4885422580810224, -1.887526963845743),
               (0.94, 17.0, -4.7819548872180455, 6.045112781954884, 0.7112847090291323, 18.88694830479505, 1.9516818207772328, 1.5728010253199605),
               (0.94, 17.0, 6.045112781954884, -6.586466165413534, 3.052619793271404, 30.193784893097025, -2.1551974669841765, -2.531455071934998),
               (0.5, 80.0, -4.7819548872180455, 6.045112781954884, -5.922152471670072, 20.379012475150994, 2.704531654613213, 0.9394488119114899),
               (0.5, 80.0, 6.045112781954884, -6.586466165413534, 36.263412372902955, 116.11214310230238, -6.127403525847198, -11.587612094178233))
        for (a, inc, α, β, k1, k2, ex, ey) in ref
            θo = deg2rad(inc)
            m = PolarizedDisk(EquatorialDisk(a))
            q = surface_prep(α, β, θo, a; ro = 1e15)
            l = surface_landing(q, π / 2, first(image_crossings(q, π / 2, 0)))
            @test l.valid
            # AART's gauge: the polarization vector with f_t = 1
            u = disk_velocity(l.r, a); uz = zamo_components(u, l.r, l.θ, a)
            pcon, pr = rest_photon(l, a, uz)
            Br = rest_components(SVector(0.0, 1.0, 0.0, 0.0), uz)
            k = SVector(pr[2], pr[3], pr[4]) / pr[1]
            f = SVector(k[2] * Br[4] - k[3] * Br[3], k[3] * Br[2] - k[1] * Br[4], k[1] * Br[3] - k[2] * Br[2])
            fbl = zamo_vector(rest_components(SVector(0.0, f...), uz; inverse = true), l.r, l.θ, a)
            gcov, _ = kerr_bl_metric(l.r, l.θ, a)
            κ = penrose_walker(l.r, l.θ, a, pcon, fbl) / (gcov * fbl)[1]
            @test real(κ) ≈ k1 rtol = 1e-9
            @test imag(κ) ≈ k2 rtol = 1e-9
            Eα, Eβ = screen_polarization(κ, α, β, θo, a)
            @test Eα ≈ ex rtol = 1e-9
            @test Eβ ≈ ey rtol = 1e-9
            # the Stokes vector of the material is along that EVPA with |P| = I
            I, Q, U = polarized_emission(m, l, α, β, θo)
            @test I == intensity(m.disk, l)
            @test hypot(Q, U) ≈ I rtol = 1e-12
            @test abs(mod(atan(U, Q) / 2 - atan(-Eα, Eβ) + π / 2, π) - π / 2) < 1e-9   # mod π
        end
    end

    @testset "intensity, polarized intensity, vacuum" begin
        a = 0.94; θo = deg2rad(17.0)
        m = readme_surface(a, deg2rad(75.0))
        q = surface_prep(5.0, 2.0, θo, a; ro = Inf)
        l = surface_landing(q, m.theta_s, first(image_crossings(q, m.theta_s, 0)))
        I, Q, U = polarized_emission(m, l, 5.0, 2.0, θo)
        @test I == intensity(m, l)
        @test hypot(Q, U) ≈ I rtol = 1e-13
        bad = Landing{Float64}(NaN, NaN, NaN, 1, false, 1.0, 2.0, π / 2, 1)
        @test polarized_emission(m, bad, 5.0, 2.0, θo) === (0.0, 0.0, 0.0)
        @test intensity(m, bad) === 0.0
        @test polarized_emission(PolarizedDisk(EquatorialDisk(a)), bad, 5.0, 2.0, θo) === (0.0, 0.0, 0.0)
        @test polarized_emission(EquatorialDisk(a), l, 5.0, 2.0, θo) ===
              (intensity(EquatorialDisk(a), l), 0.0, 0.0)
        @test surface_angle(EquatorialDisk(a)) === π / 2 && surface_angle(m) === deg2rad(75.0)
    end

    @testset "hot path" begin
        a = 0.94; θo = deg2rad(17.0)
        m = readme_surface(a, deg2rad(75.0)); d = PolarizedDisk(EquatorialDisk(a))
        q = surface_prep(5.0, 2.0, θo, a; ro = Inf)
        l = surface_landing(q, m.theta_s, 1)
        @test allocs(polarized_emission, m, l, 5.0, 2.0, θo) == 0
        @test allocs(polarized_emission, d, l, 5.0, 2.0, θo) == 0
        @test allocs(surface_landing, q, m.theta_s, 1) == 0
        @test isbits(m) && isbits(d) && Decks.device_isbits(m) && Decks.device_isbits(d)
        rt = Base.return_types(polarized_emission, (typeof(m), typeof(l), Float64, Float64, Float64))
        @test only(rt) == NTuple{3,Float64}
        # derivatives through a polarized landing
        I_of_a(x) = order_stokes(readme_surface(x, deg2rad(75.0)), 5.0, 2.0, θo, x, 0)[2]
        h = 1e-6
        @test ForwardDiff.derivative(I_of_a, a) ≈ (I_of_a(a + h) - I_of_a(a - h)) / 2h rtol = 1e-5
    end
end
