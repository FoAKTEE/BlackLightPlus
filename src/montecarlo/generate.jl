# Monte Carlo — generate.jl
#
# Superphoton generation from an `EmissionZone` (after gpumonty kernels.cu
# `sample_zone_photon`), pure state-threading so the same code runs on CPU and
# device: (a) propose nu log-uniformly, accept with (int_jnu_zone/w)/dnmax so
# each packet carries weight w = weight_at(nu); (b) propose cos(theta)
# isotropically about b-hat, accept with j_nu(sin theta)/j_nu(pi/2); (c) build
# the tetrad-frame null wavevector with k^(1) along b-hat = e_(1); (d) map to
# coordinates through the (u, b-hat) tetrad and lower to the covariant p_mu.

# Safety cap on the rejection loops so a pathological zone can never hang a warp
# (the acceptance is O(1) for any physical plasma, so this is never reached in
# practice). Mirrors rng.jl `_GAMMA_MAX_ITER`.
const _MC_REJECT_MAX_ITER = 100_000

# Frequency-rejection accept numerator: the exact Simpson pitch integral
# (`nothing`, the default) or the prebuilt log-nu table (`IntJnuTable`,
# emission_tables.jl) when tabulated generation is opted in. Selected by
# dispatch, not a runtime flag, so the default specialization compiles to the
# identical body (bitwise-unchanged streams).
@inline _gen_int_jnu(::Nothing, n_e, theta_e, B, nu) = int_jnu_zone(n_e, theta_e, B, nu)
@inline _gen_int_jnu(gt::IntJnuTable, n_e, theta_e, B, nu) = int_jnu_at(nu, gt)

"""
    sample_zone_photon(zone, wgt, dnmax, rng[, gt]) -> (Superphoton, MCRng)

Generate one superphoton from `zone` using the prebuilt weight table `wgt` and the
zone's frequency-rejection envelope `dnmax` (from `zone_nsuperphotons`). Returns
the packet and the counter-advanced RNG. Allocation-free; pure over `(zone, wgt,
dnmax, rng)`. Passing an `IntJnuTable` as `gt` switches the frequency-rejection
numerator from the live pitch integral to the tabulated interpolant (opt-in speed
path; `dnmax` must then be the matching `tabulated_dnmax` envelope so the
acceptance stays <= 1).
"""
@inline function sample_zone_photon(zone::EmissionZone, wgt::WeightTable,
                                    dnmax::Real, rng::MCRng,
                                    gt::Union{Nothing,IntJnuTable} = nothing)
    ne = zone.n_e; te = zone.theta_e; B = zone.B

    # (a) frequency rejection — accept prob (int_jnu_zone/w)/dnmax
    nu = 0.0
    w = 0.0
    @inbounds for _ in 1:_MC_REJECT_MAX_ITER
        u1, rng = mc_uniform(rng)
        nu = exp(u1 * (wgt.lnu_max - wgt.lnu_min) + wgt.lnu_min)
        w = weight_at(nu, wgt)
        u2, rng = mc_uniform(rng)
        accept = (_gen_int_jnu(gt, ne, te, B, nu) / (w + _WEIGHT_FLOOR)) / dnmax
        u2 <= accept && break
    end

    # (b) pitch rejection — accept prob j_nu(sin theta)/j_nu(pi/2)
    jm = jmax(ne, te, B, nu)
    cth = 0.0
    @inbounds for _ in 1:_MC_REJECT_MAX_ITER
        u3, rng = mc_uniform(rng)
        cth = 2.0 * u3 - 1.0
        sth = sqrt(max(0.0, 1.0 - cth * cth))
        u4, rng = mc_uniform(rng)
        u4 <= j_nu(ne, te, B, nu, sth) / jm && break
    end
    u5, rng = mc_uniform(rng)
    phi = 2.0 * π * u5
    sth = sqrt(max(0.0, 1.0 - cth * cth))

    # (c) tetrad-frame null wavevector, k^(1) aligned with b-hat = e_(1)
    eps = nu * HPL / (ME * CL * CL)
    k_tet = SVector(eps, eps * cth, eps * sth * cos(phi), eps * sth * sin(phi))

    # (d) tetrad -> coordinate; k^mu = Sum_a e_(a)^mu k^(a) = transpose(E) * k_tet
    E = emission_tetrad(zone.ucon, zone.ucov, zone.bcon, zone.gcov, zone.gcon)
    kcon = transpose(E) * k_tet
    pcov = zone.gcov * kcon             # BlackLightPlus carries covariant momentum

    sp = Superphoton(zone.x, pcov, w;
                     tau_abs = 0.0, tau_scatt = 0.0, nscatt = 0,
                     x1i = zone.x[2], x2i = zone.x[3],
                     ne0 = ne, thetae0 = te, b0 = B)
    return sp, rng
end

"""
    sample_zone_photon(zone, wgt, rng) -> (Superphoton, MCRng)

Convenience method that derives the rejection envelope `dnmax` from the zone
itself (via `zone_nsuperphotons`) before sampling. Prefer the 4-argument form in a
hot loop and pass a `dnmax` computed once per zone.
"""
@inline function sample_zone_photon(zone::EmissionZone, wgt::WeightTable, rng::MCRng)
    _, dnmax = zone_nsuperphotons(zone, wgt)
    return sample_zone_photon(zone, wgt, dnmax, rng)
end
