# Monte Carlo — spectrum.jl
#
# Escaped superphotons are binned in (exit angle, energy) and turned into
# nu L_nu, in the grmonty/gpumonty layout: log-frequency bins x the exit polar
# angle folded about the equator. Each packet contributes `w` and `w E`
# (E = -p_t, m_e c^2 units) plus a raw count. Per angle bin, nu L_nu =
# (m_e c^2)(4 pi / dOmega)(1 / dln nu) Sum w_j E_j, dOmega = 2 * 2 pi
# (cos theta_i - cos theta_f) the folded bin solid angle (bins sum to 4 pi);
# the angle-summed total drops 4 pi / dOmega and is compared to `analytic_nuLnu`.

# ============================================================ Spectrum

"""
    Spectrum

Binned escaped-superphoton accumulator. `count`/`sumw`/`sumwE` are
`n_thbins * n_ebins` matrices holding, per cell, the raw superphoton count,
`Sum w` (dNdlE) and `Sum w E` (dEdlE, `E` in m_e c^2 units). Frequency bin `i`
(1-based) is centred at `exp(lnu_min + (i - 0.5) dlnu)`; angle bin `j` spans
`[(j-1) dth, j dth]` of the equator-folded polar angle, `dth = (pi/2)/n_thbins`.
Host storage (mutable `Matrix`), reduced across threads after the run.
"""
struct Spectrum
    nu_min::Float64
    nu_max::Float64
    lnu_min::Float64
    dlnu::Float64
    n_ebins::Int
    n_thbins::Int
    count::Matrix{Int}
    sumw::Matrix{Float64}
    sumwE::Matrix{Float64}
end

"Keyword constructor: zeroed accumulators on a log-nu * folded-theta grid."
function Spectrum(; nu_min::Real = _MC_NU_MIN_DEFAULT, nu_max::Real = _MC_NU_MAX_DEFAULT,
                  n_ebins::Integer = 200, n_thbins::Integer = 6)
    lnu_min = log(nu_min)
    lnu_max = log(nu_max)
    dlnu = (lnu_max - lnu_min) / n_ebins
    return Spectrum(Float64(nu_min), Float64(nu_max), lnu_min, dlnu,
                    Int(n_ebins), Int(n_thbins),
                    zeros(Int, n_thbins, n_ebins),
                    zeros(Float64, n_thbins, n_ebins),
                    zeros(Float64, n_thbins, n_ebins))
end

# Folded polar-angle bin width.
@inline _dth(spec::Spectrum) = (0.5 * π) / spec.n_thbins

"""
    _dOmega(spec, j) -> Float64

Solid angle [sr] of folded polar-angle bin `j` (1-based): `2 * 2 pi (cos theta_i -
cos theta_f)` over `[(j-1) dth, j dth]`, the leading factor 2 mirroring the
southern hemisphere. Summed over `j = 1 … n_thbins` this is exactly `4 pi`.
"""
@inline function _dOmega(spec::Spectrum, j::Integer)
    dth = _dth(spec)
    thi = (j - 1) * dth
    thf = j * dth
    return 2.0 * 2.0 * π * (cos(thi) - cos(thf))
end

"""
    record!(spec, sph) -> spec

Bin the escaped packet `sph` by exit angle and energy, adding its count, `w` and
`w E` to the cell. `E = -p_t` (m_e c^2 units) fixes `nu = E m_e c^2 / h` and hence
the frequency bin; the exit polar angle is taken about the x3 (z) axis and folded
about the equator (`mu = |k^3| / |k_spatial|`, valid in the flat escape region
where `k^i = p_i`). Packets outside the frequency grid are dropped.
"""
@inline function record!(spec::Spectrum, sph::Superphoton)
    E = sph.E
    E > 0.0 || return spec
    nu = E * ME * CL * CL / HPL
    ie = floor(Int, (log(nu) - spec.lnu_min) / spec.dlnu) + 1
    (ie < 1 || ie > spec.n_ebins) && return spec

    p = sph.p                                    # flat escape: k^i = p_i
    ps = sqrt(p[2] * p[2] + p[3] * p[3] + p[4] * p[4])
    mu = ps > 0.0 ? abs(p[4]) / ps : 0.0         # |cos theta| about z, folded
    thf = acos(clamp(mu, 0.0, 1.0))              # in [0, pi/2]
    ith = clamp(floor(Int, thf / _dth(spec)) + 1, 1, spec.n_thbins)

    @inbounds begin
        spec.count[ith, ie] += 1
        spec.sumw[ith, ie]  += sph.w
        spec.sumwE[ith, ie] += sph.w * E
    end
    return spec
end

"""
    record_energy!(spec, sph) -> spec

Bin the escaped packet `sph` by energy only into row 1 (`nu = E m_e c^2 / h`),
ignoring direction. Used for the GRMHD per-inclination spectra, where the exit
angle is computed externally from the asymptotic geodesic direction and each
inclination band owns a single-row `Spectrum`. Off-grid packets are dropped.
"""
@inline function record_energy!(spec::Spectrum, sph::Superphoton)
    E = sph.E
    E > 0.0 || return spec
    nu = E * ME * CL * CL / HPL
    ie = floor(Int, (log(nu) - spec.lnu_min) / spec.dlnu) + 1
    (ie < 1 || ie > spec.n_ebins) && return spec
    @inbounds begin
        spec.count[1, ie] += 1
        spec.sumw[1, ie]  += sph.w
        spec.sumwE[1, ie] += sph.w * E
    end
    return spec
end

"""
    nuLnu_solid(spec, dOmega) -> (nu, L, count)

Isotropic-equivalent `nu L_nu(nu) = (m_e c^2)(4 pi / dOmega)(1 / dln nu) Sum w E`
[erg s^-1] for a single-row (angle-selected) spectrum whose escaped photons
subtend solid angle `dOmega` [sr] — the grmonty `report_spectrum`
per-viewing-angle convention (the equivalent all-sky luminosity a distant
observer at that inclination infers). `count` is the per-bin raw photon count.
"""
function nuLnu_solid(spec::Spectrum, dOmega::Real)
    nu, L, cnt = nuLnu_total(spec)
    f = 4.0 * π / dOmega
    @inbounds for i in eachindex(L)
        L[i] *= f
    end
    return nu, L, cnt
end

"""
    merge_spectra!(dst, src) -> dst

Add `src`'s accumulators into `dst` (grid-compatible). Used to reduce per-thread
spectra after a threaded run.
"""
function merge_spectra!(dst::Spectrum, src::Spectrum)
    @assert dst.n_ebins == src.n_ebins && dst.n_thbins == src.n_thbins "grid mismatch"
    @inbounds for k in eachindex(dst.count)
        dst.count[k] += src.count[k]
        dst.sumw[k]  += src.sumw[k]
        dst.sumwE[k] += src.sumwE[k]
    end
    return dst
end

# ============================================================ nu L_nu accessors

"Frequency bin centres [Hz]."
@inline nu_centers(spec::Spectrum) =
    [exp(spec.lnu_min + (i - 0.5) * spec.dlnu) for i in 1:spec.n_ebins]

"""
    nuLnu(spec) -> (nu, L)

Per-angle-bin `nu L_nu` [erg s^-1]: `L[j, i] = (m_e c^2)(4 pi / dOmega_j)(1 / dln
nu) Sum w E` (gpumonty `report_spectrum` convention). `nu` are the bin centres.
"""
function nuLnu(spec::Spectrum)
    nu = nu_centers(spec)
    L = zeros(Float64, spec.n_thbins, spec.n_ebins)
    pref = ME * CL * CL / spec.dlnu
    @inbounds for j in 1:spec.n_thbins
        f = pref * 4.0 * π / _dOmega(spec, j)
        for i in 1:spec.n_ebins
            L[j, i] = f * spec.sumwE[j, i]
        end
    end
    return nu, L
end

"""
    nuLnu_total(spec) -> (nu, L, count)

Angle-summed (total) `nu L_nu(nu) = (m_e c^2)(1 / dln nu) Sum_{all theta} w E` [erg
s^-1], with the per-bin raw superphoton `count` (summed over angle) for the
population cut. This is `Sum_theta nuLnu(theta) dOmega / (4 pi)` — the convention
matching `analytic_nuLnu`.
"""
function nuLnu_total(spec::Spectrum)
    nu = nu_centers(spec)
    L = zeros(Float64, spec.n_ebins)
    cnt = zeros(Int, spec.n_ebins)
    pref = ME * CL * CL / spec.dlnu
    @inbounds for i in 1:spec.n_ebins
        s = 0.0
        c = 0
        for j in 1:spec.n_thbins
            s += spec.sumwE[j, i]
            c += spec.count[j, i]
        end
        L[i] = pref * s
        cnt[i] = c
    end
    return nu, L, cnt
end

"""
    total_power(spec) -> Float64

Bolometric recorded power `Sum_all w h nu = (m_e c^2) Sum_all w E` [erg s^-1] over
every cell — the Monte Carlo estimate of the sphere's frequency-integrated
luminosity (optically thin: all emitted energy escapes).
"""
@inline total_power(spec::Spectrum) = ME * CL * CL * sum(spec.sumwE)

"""
    total_weight(spec) -> Float64

Total recorded superphoton weight `Sum_all w` over every cell (the packet-number
content of the spectrum, used to normalise per-order populations).
"""
@inline total_weight(spec::Spectrum) = sum(spec.sumw)

"""
    mean_energy(spec) -> Float64

Weight-averaged escaped-photon energy `<E> = Sum w E / Sum w` [m_e c^2 units] over
all cells (`0` if the spectrum is empty). For the per-scattering-order spectra of
`mc_spectrum_scatter` this is the mean energy of that Compton bump, so the ratio
between successive orders measures the up-scattering amplification.
"""
@inline function mean_energy(spec::Spectrum)
    sw = sum(spec.sumw)
    return sw > 0.0 ? sum(spec.sumwE) / sw : 0.0
end
