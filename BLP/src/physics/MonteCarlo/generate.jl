# Superphoton generation from an emission zone (gpumonty `kernels.cu`
# `sample_zone_photon`), threading the stream: (a) propose ν log-uniformly and accept with
# probability (∫j_ν dΩ / w(ν))/dnmax, so the packet carries w(ν); (b) propose cos θ
# uniformly about b̂ and accept with j_ν(θ)/j_ν(π/2); (c) build the tetrad-frame null
# wave vector with k^(1) along b̂ = e_(1); (d) map it to coordinates through the emission
# tetrad and lower it.

"Cap of each rejection loop of generation; the acceptance is O(1) for any emitting zone."
const REJECT_MAX_ITER = 100_000

"""
    GenerationTables{W,G}(weights, dnmax[, int_jnu = nothing])

What a zone draws its superphotons from: the weight table `weights`, the zone's
frequency-rejection envelope `dnmax` (from `zone_nsuperphotons`, or `tabulated_dnmax`
with a table), and `int_jnu`, `nothing` to evaluate `int_jnu_zone` at every proposal
or the zone's `IntJnuTable` (tabulated generation, statistically equivalent but not
bitwise). `isbits` after `Adapt.adapt` to a device.
"""
struct GenerationTables{W<:WeightTable,G<:Union{Nothing,IntJnuTable}}
    weights::W
    dnmax::Float64
    int_jnu::G
end
GenerationTables(weights::WeightTable, dnmax::Real) =
    GenerationTables(weights, Float64(dnmax), nothing)
Adapt.@adapt_structure GenerationTables

# Numerator of the frequency acceptance: the Simpson integral or the tabulated one.
@inline gen_int_jnu(::Nothing, n_e, theta_e, B, nu) = int_jnu_zone(n_e, theta_e, B, nu)
@inline gen_int_jnu(gt::IntJnuTable, n_e, theta_e, B, nu) = int_jnu_at(nu, gt)

"""
    generate(zone::EmissionZone, tables::GenerationTables, rng::MCRng) -> (Superphoton, MCRng)

One superphoton emitted by `zone` and the advanced stream. The frequency is drawn by
rejection against `tables` (weight `w(ν)` of the packet), the pitch angle to the field
by rejection against `jmax`, the azimuth uniformly; the momentum is null in the
emission tetrad of the zone and covariant in coordinates, the weight is `w(ν)`, the
optical depths and scattering count are zero, and the emission diagnostics are the
zone's `x¹`, `x²`, `n_e`, `Θₑ`, `B`. Each rejection loop stops after `REJECT_MAX_ITER`
proposals with the last one. Pure and allocation-free.
"""
@inline function generate(zone::EmissionZone, tables::GenerationTables, rng::MCRng)
    wgt = tables.weights
    dnmax = tables.dnmax
    gt = tables.int_jnu
    ne = zone.n_e; te = zone.theta_e; B = zone.B

    nu = 0.0
    w = 0.0
    @inbounds for _ in 1:REJECT_MAX_ITER
        u1, rng = mc_uniform(rng)
        nu = exp(u1 * (wgt.lnu_max - wgt.lnu_min) + wgt.lnu_min)
        w = weight_at(nu, wgt)
        u2, rng = mc_uniform(rng)
        accept = (gen_int_jnu(gt, ne, te, B, nu) / (w + WEIGHT_FLOOR)) / dnmax
        u2 <= accept && break
    end

    jm = jmax(ne, te, B, nu)
    cth = 0.0
    @inbounds for _ in 1:REJECT_MAX_ITER
        u3, rng = mc_uniform(rng)
        cth = 2.0 * u3 - 1.0
        sth = sqrt(max(0.0, 1.0 - cth * cth))
        u4, rng = mc_uniform(rng)
        u4 <= j_nu(ne, te, B, nu, sth) / jm && break
    end
    u5, rng = mc_uniform(rng)
    phi = 2.0 * π * u5
    sth = sqrt(max(0.0, 1.0 - cth * cth))

    eps = nu * HPL / (ME * CL * CL)
    k_tet = SVector(eps, eps * cth, eps * sth * cos(phi), eps * sth * sin(phi))

    E = emission_tetrad(zone.ucon, zone.ucov, zone.bcon, zone.gcov, zone.gcon)
    kcon = transpose(E) * k_tet
    pcov = zone.gcov * kcon

    sp = Superphoton(zone.x, pcov, w;
                     tau_abs = 0.0, tau_scatt = 0.0, nscatt = 0,
                     x1i = zone.x[2], x2i = zone.x[3],
                     ne0 = ne, thetae0 = te, b0 = B)
    return sp, rng
end

"""
    generate(sphere::UniformSphere, tables::GenerationTables, rng::MCRng) -> (Superphoton, MCRng)

One superphoton of the uniform sphere: the emission point is uniform in the ball,
`r = R u^{1/3}` along an isotropic direction (three draws), and the photon is
generated from the sphere's zone there, `emission_zone(sphere, x, g)` with `g` the
Minkowski metric pair, by the zone method.
"""
@inline function generate(s::UniformSphere, tables::GenerationTables, rng::MCRng)
    u, rng = mc_uniform(rng)
    r = s.R * cbrt(u)
    dir, rng = sample_direction(rng)
    x = SVector(0.0, r * dir[1], r * dir[2], r * dir[3])
    zone = emission_zone(s, x, metric_and_inverse(Minkowski(), x))
    return generate(zone, tables, rng)
end

"""
    table_zone(sphere::UniformSphere) -> EmissionZone

The zone that stands for the whole sphere in its generation tables: the sphere's zone
at the origin, whose volume is that of the ball.
"""
function table_zone(s::UniformSphere)
    x0 = SVector(0.0, 0.0, 0.0, 0.0)
    return emission_zone(s, x0, metric_and_inverse(Minkowski(), x0))
end
