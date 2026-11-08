# m87_scaling.jl — the M87* scaling of the paper's GRMHD figures and the flux calibration
# of their density scale (module M87Scaling; included by the generators, never run alone).
#
# Every GRMHD run is scaled to M87* (Event Horizon Telescope Collaboration 2019, papers V
# and VI): black-hole mass 6.5e9 Msun, for each black hole of an equal-mass binary, whose
# code mass unit (the total mass) is then 1.3e10 Msun; distance 16.8 Mpc; 230 GHz. The
# density scale of a run (`mass_scale_cgs`, the density unit being mass_scale / (GM/c²)³)
# is the one at which its unpolarized 230 GHz image seen from inclination 17° holds
# 0.5 Jy: `m87_calibrate` root-finds it on that image, with the run's plasma, metric,
# camera, field of view and pixel grid, at the step divisor C = 320. M87_DENSITY_SCALE
# records the scale of every run, its calibration image and the uncertainty of its 0.5 Jy,
# as m87_calibrate.jl gives them.
module M87Scaling

using Printf
using StaticArrays
using BlackLightPlus
using BlackLightPlus.Numerics: Units
using BlackLightPlus.Execution: resolve, foreach_item!, CoordList
using BlackLightPlus.Plans: transfer_params
using BlackLightPlus.Products: imaging_program

export M87_MSUN_G, M87_MASS_MSUN, M87_MASS_G, M87_DISTANCE_MPC, M87_DISTANCE_CM,
       M87_FREQUENCY_HZ, M87_FLUX_JY, M87_INCLINATION_DEG, M87_C, M87_FLUX_RTOL,
       M87_DENSITY_SCALE, m87_density_scale, m87_pixel_sr, m87_flux_jy, m87_flux_ladder,
       m87_ladder_root, m87_calibrate

const M87_MSUN_G = 1.98847e33                     # g, the nominal solar mass (IAU 2015)
const M87_MASS_MSUN = 6.5e9                       # each black hole
const M87_MASS_G = M87_MASS_MSUN * M87_MSUN_G
const M87_DISTANCE_MPC = 16.8
const M87_DISTANCE_CM = M87_DISTANCE_MPC * 1.0e6 * 3.0856775814913673e18
const M87_FREQUENCY_HZ = 230.0e9
const M87_FLUX_JY = 0.5                           # 230 GHz flux density
const M87_INCLINATION_DEG = 17.0                  # inclination of the calibration image
const M87_C = 320.0                               # step divisor of the calibration image
const M87_FLUX_RTOL = 1.0e-3                      # |F/0.5 Jy - 1| of the root, on the calibration image

"""
    M87_DENSITY_SCALE

The calibrated density scale of every GRMHD run, by snapshot number: `mass_scale_g`, the
code mass unit `m_bh_msun` (the total mass of a binary), the calibration image
(inclination 17°, full field of view `fov` in GM/c², `res` pixels per side, step divisor
`C`) with the flux `flux_jy` it holds there, and `flux_rtol`, the bound on |F/0.5 Jy - 1|
at that scale on any pixel grid of the field.

The root holds the calibration image to M87_FLUX_RTOL = 1e-3. For 00999 the flux of the
field changes by 4e-6 from 768 to 1536 pixels per side, so `flux_rtol` is that tolerance.
For the binaries the flux depends on the pixel grid: over nine grids of 480–1920 pixels per
side it scatters by 0.13 % (00455) and 0.08 % (00122) rms without a trend, the quadrature
scatter of the compact bright gas next to the black holes, and lies within 3.3e-3 and
2.9e-3 of 0.5 Jy on every grid. `flux_rtol` is that pixel scatter; with dlnF/dlnms ≈ 2 it
fixes the binary scales to 1.7e-3 and 1.5e-3. Doubling C to 640 changes the flux by at
most 3e-4.
"""
const M87_DENSITY_SCALE = Dict{String,NamedTuple}(
    # Kerr accretion flow, AthenaK 00999 on Kerr a = 0.9, R_high = 10 (fig_pol_hires.jl)
    "00999" => (mass_scale_g = 2.272651712e23, m_bh_msun = 6.5e9, fov = 30.0, res = 768,
                C = 320.0, flux_jy = 0.5000296497, flux_rtol = 1.0e-3),
    # binary GRMHD, AthenaK 00455, separation 25 M, R_high = 40 (fig_eht_bbh.jl);
    # flux_rtol: the 800-pixel grid, 0.498380 Jy
    "00455" => (mass_scale_g = 1.589253825e28, m_bh_msun = 1.3e10, fov = 56.0, res = 960,
                C = 320.0, flux_jy = 0.5000122263, flux_rtol = 3.3e-3),
    # binary GRMHD, AthenaK 00122, separation 16 M, R_high = 40 (fig_eht_bbh.jl);
    # flux_rtol: the 480-pixel grid, 0.501443 Jy
    "00122" => (mass_scale_g = 7.007445187e20, m_bh_msun = 1.3e10, fov = 56.0, res = 960,
                C = 320.0, flux_jy = 0.5000071296, flux_rtol = 2.9e-3),
)

"The calibrated density scale (g) of run `run` (snapshot number, e.g. \"00455\")."
function m87_density_scale(run::AbstractString)
    haskey(M87_DENSITY_SCALE, run) ||
        error("m87_density_scale: run $run has no calibrated scale in m87_scaling.jl")
    return M87_DENSITY_SCALE[run].mass_scale_g
end

"Solid angle (sr) of one pixel of a `res`² image of full field `fov` (GM/c² of `m_bh_g`) at 16.8 Mpc."
m87_pixel_sr(fov, res, m_bh_g) = (fov / res * Units(m_bh_g, 1.0).L_unit / M87_DISTANCE_CM)^2

"Flux density (Jy) of the square intensity image `I` (cgs) of full field `fov` (GM/c² of `m_bh_g`)."
m87_flux_jy(I, fov, m_bh_g) = sum(I) * m87_pixel_sr(fov, size(I, 1), m_bh_g) / 1.0e-23

"Pixel index subset on a stride-s sub-lattice (flux estimator weight s^2)."
function stride_idx(res, s)
    o = (s + 1) ÷ 2
    return [(ia - 1) * res + ib for ia in o:s:res for ib in o:s:res]
end

"""
    m87_flux_ladder(plan, mss; stride = 1, backend = :cpu) -> Vector{Float64}

The unpolarized flux density (Jy) of the image of `plan` for every mass scale in `mss`,
one march per ray carrying every lane: `variants` on the full frame (stride 1), else the
Stokes I imaging program with the ladder's parameter sets over the pixel centres of the
stride sub-lattice, weighted by stride².
"""
function m87_flux_ladder(plan, mss; stride = 1, backend = :cpu)
    K = length(mss)
    res = plan.camera.npix
    idx = stride_idx(res, stride)
    acc = zeros(K, length(idx))
    if stride == 1
        v = variants(plan, collect(Float64, mss); backend = backend)
        for k in 1:K, n in eachindex(idx)
            acc[k, n] = v.images[k][idx[n]]
        end
    else
        c = plan.pixels.centres
        gen = CoordList(Float64[c[(i - 1) ÷ res + 1] for i in idx],
                        Float64[c[(i - 1) % res + 1] for i in idx])
        params = ntuple(k -> transfer_params(plan.config; mass_scale = Float64(mss[k])), K)
        out = Vector{SVector{K,Float64}}(undef, length(idx))
        foreach_item!(resolve(backend), imaging_program(plan; mode = StokesI(), params, gen),
                      eachindex(out); outs = (out,))
        for k in 1:K, n in eachindex(idx)
            acc[k, n] = out[n][k]
        end
    end
    return vec(sum(acc; dims = 2)) .*
           (stride^2 * m87_pixel_sr(plan.camera.fov, res, plan.config.m_bh_cgs) / 1.0e-23)
end

"Log-log interpolated root of F(ms) = `target` on a ladder and the local slope dlnF/dlnms, or nothing if unbracketed."
function m87_ladder_root(mss, F, target)
    lm = log.(mss); lf = log.(max.(F, 1e-300)); lt = log(target)
    for k in 1:length(mss)-1
        if (lf[k] - lt) * (lf[k+1] - lt) <= 0 && F[k] > 0 && F[k+1] > 0
            w = (lt - lf[k]) / (lf[k+1] - lf[k])
            return exp(lm[k] + w * (lm[k+1] - lm[k])), (lf[k+1] - lf[k]) / (lm[k+1] - lm[k])
        end
    end
    return nothing
end

_ladder_line(tag, mss, F) =
    @printf("    ladder %s: %s\n", tag, join([@sprintf("%.4e:%.5g", m, f) for (m, f) in zip(mss, F)], " "))

"""
    m87_calibrate(plan, ms_guess; target = M87_FLUX_JY, rtol = M87_FLUX_RTOL,
                  backend = :cpu, verbose = true)
        -> (ms, flux_jy, slope, converged, monotonic, ladders, wall)

The density scale `ms` (g) at which the unpolarized image of `plan` (a plan of any mode
and mass scale; its camera, plasma, metric and march define the image) holds `target` Jy
at 16.8 Mpc. Ladders of mass scales (`m87_flux_ladder`) bracket the root: 9 lanes 1 dex
apart about `ms_guess` on the stride-4 pixel sub-lattice, recentred up to 4 times until
bracketed; 5 lanes 0.1 dex apart about the log-log interpolated root on every pixel; then,
on every pixel, 3 lanes 0.004 dex apart about the root, moved to the root they bracket
(or by a secant step with their slope) until the flux at the centre lane is within
`rtol` of `target` (at most 4 rounds; `converged` says whether it is). `flux_jy` is that
centre-lane flux, `slope` dlnF/dlnms there, and `monotonic` whether the flux increases
along the fine ladder and along the coarse one up to its first lane above the target.
"""
function m87_calibrate(plan, ms_guess; target = M87_FLUX_JY, rtol = M87_FLUX_RTOL,
                       backend = :cpu, verbose = true)
    t0 = time()
    ladders = Any[]
    lad(stride, mss, F) = (push!(ladders, Dict("stride" => stride, "mass_scale" => collect(mss),
                                               "flux_Jy" => F));
                           verbose && _ladder_line("s$stride", mss, F))
    # coarse: stride 4, 1 dex spacing, recentred until bracketed
    center = ms_guess
    root = nothing
    Fc = Float64[]
    for pass in 1:4
        mss = center .* 10.0 .^ (-4.0:1.0:4.0)
        Fc = m87_flux_ladder(plan, mss; stride = 4, backend)
        lad(4, mss, Fc)
        root = m87_ladder_root(mss, Fc, target)
        root !== nothing && break
        center = Fc[end] < target ? mss[end] * 1e4 : mss[1] * 1e-4
    end
    root === nothing && error("m87_calibrate: $(target) Jy not bracketed")
    # fine: every pixel, +-0.2 dex
    mss = root[1] .* 10.0 .^ (-0.2:0.1:0.2)
    Ff = m87_flux_ladder(plan, mss; stride = 1, backend)
    lad(1, mss, Ff)
    r2 = m87_ladder_root(mss, Ff, target)
    ms, slope = r2 === nothing ? root : r2
    kb = findfirst(>=(target), Fc)
    mono = all(diff(Ff) .> 0) && kb !== nothing && all(diff(Fc[1:kb]) .> 0)
    # refine: every pixel, 3 lanes 0.004 dex apart about the root
    F = NaN; converged = false
    for round in 1:4
        mss = ms .* 10.0 .^ (-0.004:0.004:0.004)
        Fr = m87_flux_ladder(plan, mss; stride = 1, backend)
        lad(1, mss, Fr)
        F = Fr[2]
        slope = log(Fr[3] / Fr[1]) / log(mss[3] / mss[1])
        if abs(F / target - 1) <= rtol
            converged = true
            break
        end
        r3 = m87_ladder_root(mss, Fr, target)
        ms = r3 === nothing ? ms * exp(log(target / F) / slope) : r3[1]
    end
    return (ms = ms, flux_jy = F, slope = slope, converged = converged, monotonic = mono,
            ladders = ladders, wall = time() - t0)
end

end # module M87Scaling
