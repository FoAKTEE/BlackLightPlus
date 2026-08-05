# Monte Carlo — hotcross.jl
#
# Thermally-averaged (hot) Compton cross section sigma_h(eps, Theta_e) — the
# scattering opacity: the Klein-Nishina total sigma_KN Doppler-boosted into
# each electron's rest frame and averaged over the Maxwell-Juttner distribution
# and collision angle, sigma_h = (1/n_e) int d^3p (dn/d^3p)(1 - mu_e beta_e)
# sigma_KN(eps_e), eps_e = eps gamma_e (1 - mu_e beta_e) (Wienke 1985; Dolence
# et al. 2009; gpumonty hotcross.cu). Evaluated by 2D midpoint quadrature with
# warp-uniform trip counts; production interpolates a 221x81 log-log table.

# ============================================================ grid / quadrature constants
# gpumonty config.h. Table spans (eps, Theta_e) = (MINW..MAXW) x (MINT..MAXT) on an
# (NW+1) x (NT+1) log-log grid; the quadrature steps DMUE / DGAMMAE with MAXGAMMA.

const MINW     = 1.0e-12     # min dimensionless photon energy (table lower edge)
const MAXW     = 1.0e15      # max dimensionless photon energy (table upper edge)
const MINT     = 1.0e-4      # min Theta_e (below -> cold Klein-Nishina limit)
const MAXT     = 1.0e4       # max Theta_e
const NW       = 220         # energy bins (NW+1 = 221 nodes)
const NT       = 80          # temperature bins (NT+1 = 81 nodes)
const DMUE     = 0.05        # mu_e integration step
const DGAMMAE  = 0.05        # gamma_e integration step (in units of Theta_e)
const MAXGAMMA = 12.0        # integrate to gamma_e = 1 + MAXGAMMA * Theta_e

# Warp-uniform midpoint-rule node counts. Both are Theta_e-INDEPENDENT: the mu grid
# spans [-1,1] (width 2) so 2/DMUE = 40 nodes; the gamma grid spans
# MAXGAMMA*Theta_e with step DGAMMAE*Theta_e so MAXGAMMA/DGAMMAE = 240 nodes.
const _N_MUE    = 40         # round(Int, 2.0 / DMUE)
const _N_GAMMAE = 240        # round(Int, MAXGAMMA / DGAMMAE)

# ============================================================ Klein-Nishina total

"""
    kn_total(eps_e) -> Float64

Klein-Nishina TOTAL cross section normalised by the Thomson value,
`sigma_KN(eps_e) / sigma_T`, for a photon of dimensionless energy `eps_e` in the
electron rest frame. Uses the numerically stable low-energy expansion `1 - 2 eps_e`
for `eps_e < 1e-3` (avoids catastrophic cancellation in the grouped form), else the
closed KN form. `-> 1` (Thomson) as `eps_e -> 0` and decreases monotonically.
Pure / allocation-free / device-safe.
"""
@inline function kn_total(eps_e)
    w = eps_e
    if w < 1.0e-3
        return 1.0 - 2.0 * w
    end
    return 0.75 * (2.0 / (w * w) +
                   (1.0 / (2.0 * w) - (1.0 + w) / (w * w * w)) * log(1.0 + 2.0 * w) +
                   (1.0 + w) / ((1.0 + 2.0 * w) * (1.0 + 2.0 * w)))
end

"""
    boostcross(eps, mu_e, gamma_e) -> Float64

Doppler-boosted Klein-Nishina cross section (over sigma_T) for a lab-frame photon
of energy `eps` colliding with an electron of Lorentz factor `gamma_e` at collision
cosine `mu_e`. The photon energy in the electron rest frame is
`eps_e = eps gamma_e (1 - mu_e beta_e)` (`beta_e = sqrt(gamma_e^2-1)/gamma_e`) and
the result is weighted by the relative flux factor `(1 - mu_e beta_e)`.
"""
@inline function boostcross(eps, mu_e, gamma_e)
    v = sqrt(gamma_e * gamma_e - 1.0) / gamma_e
    we = eps * gamma_e * (1.0 - mu_e * v)
    return kn_total(we) * (1.0 - mu_e * v)
end

"""
    dNdgamma_e(theta_e, gamma_e) -> Float64

Normalised Maxwell-Juttner distribution `dN/dgamma_e` per unit electron number
density (integrates to 1 over `gamma_e in [1, inf)`). The normalisation uses
`K2f = K2(1/Theta_e) exp(1/Theta_e)` for `Theta_e > 1e-2`; below that the
product overflows/underflows to NaN, so the large-argument asymptotic series
`K2f ~ sqrt(pi Theta_e/2)(1 + 15 Theta_e/8 + 105 Theta_e^2/128)` (A&S 9.7.2)
is used — carried through `Theta_e^2` so the regimes agree to ~2e-7 at the switch.
"""
@inline function dNdgamma_e(theta_e, gamma_e)
    K2f = theta_e > 1.0e-2 ? bessel_k2(1.0 / theta_e) * exp(1.0 / theta_e) :
          sqrt(π * theta_e / 2.0) *
              (1.0 + theta_e * (15.0 / 8.0 + theta_e * (105.0 / 128.0)))
    return (gamma_e * sqrt(gamma_e * gamma_e - 1.0) / (theta_e * K2f)) *
           exp(-(gamma_e - 1.0) / theta_e)
end

"""
    total_compton_cross_num(eps, theta_e) -> Float64

Thermally-averaged hot Compton cross section `sigma_h(eps, Theta_e)` [cm^2], by
2D midpoint quadrature over `mu_e in [-1,1]` and `gamma_e in [1, 1 + MAXGAMMA
Theta_e]`, integrating `(1 - mu_e beta_e) sigma_KN(eps_e)` against
`dN/dgamma_e`. Cold corners short-circuit: `Theta_e < MINT && eps < MINW ->
sigma_T`; `Theta_e < MINT -> kn_total(eps) sigma_T`. Warp-uniform trip counts,
so device-safe (after gpumonty `total_compton_cross_num`).
"""
@inline function total_compton_cross_num(eps, theta_e)
    if theta_e < MINT && eps < MINW
        return SIGMA_THOMSON
    end
    if theta_e < MINT
        return kn_total(eps) * SIGMA_THOMSON
    end

    dgammae = theta_e * DGAMMAE
    cross = 0.0
    @inbounds for im in 0:(_N_MUE - 1)
        mue = -1.0 + (im + 0.5) * DMUE
        for ig in 0:(_N_GAMMAE - 1)
            gammae = 1.0 + (ig + 0.5) * dgammae
            f = 0.5 * dNdgamma_e(theta_e, gammae)
            cross += DMUE * dgammae * boostcross(eps, mue, gammae) * f
        end
    end
    return cross * SIGMA_THOMSON
end

# ============================================================ 2D log-log table

"""
    HotcrossTable

Precomputed hot-cross-section lookup table: the `(NW+1) x (NT+1)` matrix of
`log10(sigma_h [cm^2])` on the log-log `(eps, Theta_e)` grid, with
`logsigma[i+1, j+1]` at `eps = 10^(log10(MINW) + i dlw)`, `Theta_e =
10^(log10(MINT) + j dlT)`. A plain `Matrix`: a shared read-only buffer the GPU
uploads directly.
"""
struct HotcrossTable
    logsigma::Matrix{Float64}
end

"""
    load_hotcross(path) -> HotcrossTable

Parse a gpumonty `hotcross.dat` file (`i j log10(eps) log10(Theta_e) log10(sigma_h)`
per line, `(NW+1)*(NT+1)` rows) into a `HotcrossTable`. Only the integer grid
indices and the `log10(sigma_h)` column are used, so the table is laid out
independently of the file's row order.
"""
function load_hotcross(path::AbstractString)
    logsigma = Matrix{Float64}(undef, NW + 1, NT + 1)
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            logsigma[i + 1, j + 1] = parse(Float64, parts[5])
        end
    end
    return HotcrossTable(logsigma)
end

"""
    build_hotcross_table() -> HotcrossTable

Regenerate the full `(NW+1) x (NT+1)` table by evaluating
`total_compton_cross_num` at every grid node (log10 of the cm^2 result).
Host-side offline builder; production reads the cached file via
`load_hotcross`. The energy loop is threaded over independent columns, so the
fill is race-free and bit-identical regardless of thread count.
"""
function build_hotcross_table()
    logsigma = Matrix{Float64}(undef, NW + 1, NT + 1)
    dlw = log10(MAXW / MINW) / NW
    dlT = log10(MAXT / MINT) / NT
    lminw = log10(MINW)
    lmint = log10(MINT)
    Threads.@threads for i in 0:NW
        lw = lminw + i * dlw
        @inbounds for j in 0:NT
            lT = lmint + j * dlT
            logsigma[i + 1, j + 1] = log10(total_compton_cross_num(10.0^lw, 10.0^lT))
        end
    end
    return HotcrossTable(logsigma)
end

"""
    hotcross_lookup(table, eps, theta_e) -> Float64

Hot Compton cross section `sigma_h(eps, Theta_e)` [cm^2] via the log-log table.
Branch order (matches gpumonty `total_compton_cross_lkup`, ORDER MATTERS):

1. `eps*Theta_e < 1e-6` -> Thomson `sigma_T` (cold, soft),
2. `Theta_e < MINT`     -> cold Klein-Nishina `kn_total(eps) sigma_T`,
3. in-bounds            -> bilinear interpolation in `log10(eps)`, `log10(Theta_e)`,
4. otherwise            -> direct `total_compton_cross_num` fallback.

Device-safe (only reads the passed array + arithmetic).
"""
@inline hotcross_lookup(table::HotcrossTable, eps, theta_e) =
    _hotcross_lookup(table.logsigma, eps, theta_e)

@inline function _hotcross_lookup(logsigma, eps, theta_e)
    if eps * theta_e < 1.0e-6
        return SIGMA_THOMSON
    end
    if theta_e < MINT
        return kn_total(eps) * SIGMA_THOMSON
    end
    if (eps > MINW && eps < MAXW) && (theta_e > MINT && theta_e < MAXT)
        dlw = log10(MAXW / MINW) / NW
        dlT = log10(MAXT / MINT) / NT
        lminw = log10(MINW)
        lmint = log10(MINT)
        fi = (log10(eps) - lminw) / dlw
        fj = (log10(theta_e) - lmint) / dlT
        i = floor(Int, fi)
        j = floor(Int, fj)
        di = fi - i
        dj = fj - j
        @inbounds lcross =
            (1.0 - di) * (1.0 - dj) * logsigma[i + 1, j + 1] +
            di * (1.0 - dj)         * logsigma[i + 2, j + 1] +
            (1.0 - di) * dj         * logsigma[i + 1, j + 2] +
            di * dj                 * logsigma[i + 2, j + 2]
        return 10.0^lcross
    end
    return total_compton_cross_num(eps, theta_e)
end

# ============================================================ scattering opacity

"""
    kappa_es(table, nu, theta_e) -> Float64

Electron-scattering mass opacity `sigma_h / m_p` [cm^2 g^-1] at frequency `nu`
(cgs) and electron temperature `Theta_e`. The dimensionless photon energy is
`eps = HPL nu / (m_e c^2)`. Matches grmonty `kappa_es` so that
`alpha_scatt = nu kappa_es n_e m_p`.
"""
@inline function kappa_es(table::HotcrossTable, nu, theta_e)
    eps = HPL * nu / (ME * CL * CL)
    return hotcross_lookup(table, eps, theta_e) / MP
end

"""
    alpha_inv_scatt(table, nu, theta_e, n_e) -> Float64

Lorentz-INVARIANT scattering absorption coefficient `alpha_inv,s = nu n_e sigma_h`
at frequency `nu` (cgs), electron temperature `Theta_e` and number density `n_e`.
Slots into the same invariant-alpha march as the synchrotron absorptivity.
"""
@inline function alpha_inv_scatt(table::HotcrossTable, nu, theta_e, n_e)
    eps = HPL * nu / (ME * CL * CL)
    return nu * n_e * hotcross_lookup(table, eps, theta_e)
end
