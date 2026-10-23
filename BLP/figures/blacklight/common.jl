# Shared setup of the Blacklight-paper (arXiv:2203.15963) figure reproductions: the
# fiducial deck, the output directories, the flux in Jy, JSON and npy writers, and the
# sampling of plasma and transfer coefficients along stored trajectories.
#
# Fiducial system (paper section 5): the paper's thick aligned disk around a spin-0.9
# hole is replaced by the AthenaK torus dump of BLP/figures/fig_pol_hires.jl (spin 0.9,
# the same electron model: rat_low 1, rat_high 10, gamma 13/9, total-pressure split,
# sigma cut 1). The paper's free scales are kept: hole mass 4.152e6 Msun, the density mass
# scale tuned so that the fiducial image carries 2.4 Jy at 230 GHz from 8.178 kpc
# (calibrated by fig_calibrate.jl into out/blacklight/mass_scale.txt). The paper's camera:
# plane-parallel, full width 24 M, centre at r = 100 M and theta = 45 deg, 256^2 pixels,
# 230 GHz at the camera.
using Dates, LinearAlgebra, Printf, StaticArrays
using BLP
using BLP.Numerics: CL, GNEWT, MSUN, MP, ME
using BLP.Spacetime: KerrSchildCartesian, Minkowski, metric_and_inverse, ks_radius
using BLP.Matter: PlasmaPrimitives, primitives, sample, domain_radius, GridProvider,
                  pgas_from_eint, primitive_four_vectors
using BLP.Transfer: coefficients, StokesI, IQUV
using BLP.Data: write_npy, AthenaKSnapshot
using BLP.Products: Trajectories

const BL_ROOT = dirname(dirname(dirname(@__DIR__)))          # repository root
const BL_OUT = joinpath(BL_ROOT, "BLP", "figures", "out", "blacklight")
const BL_DATA_DIR = get(ENV, "BLP_TORUS_DIR", "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no")
const BL_DUMP = get(ENV, "BLP_TORUS_DUMP", "torus.mhd_w_bcc.00999.athdf")
const BL_SPIN = 0.9
const BL_MBH = 4.152e6 * MSUN
const BL_L = GNEWT * BL_MBH / CL^2
const BL_DIST = 8.178e3 * 3.0857e18                            # 8.178 kpc in cm
const BL_NU = 2.3e11
const BL_FLUX_JY = 2.4
const BL_FOV = 24.0
const BL_DCAM = 100.0
const BL_INC = 45.0
const BL_RES = 256
const BL_GIT_SHA = try
    sha = readchomp(`git -C $BL_ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $BL_ROOT status --porcelain -- BLP/src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))
outdir(name) = (d = joinpath(BL_OUT, name); mkpath(d); d)
dump_path(n::Integer) = joinpath(BL_DATA_DIR, @sprintf("torus.mhd_w_bcc.%05d.athdf", n))
dump_path() = joinpath(BL_DATA_DIR, BL_DUMP)

"the calibrated density mass scale (g), from out/blacklight/mass_scale.txt"
function mass_scale()
    f = joinpath(BL_OUT, "mass_scale.txt")
    isfile(f) || error("run fig_calibrate.jl first: $f is missing")
    return parse(Float64, strip(readline(f)))
end

"the fiducial run configuration (keyword overrides on the camera and plasma)"
function fiducial_config(; inc = BL_INC, fov = BL_FOV, res = BL_RES, dcam = BL_DCAM,
                         nu = BL_NU, mass = nothing, plasma = PlasmaConfig(),
                         max_steps = 4000, C = 80.0)
    ms = mass === nothing ? mass_scale() : mass
    pl = PlasmaConfig(mu = plasma.mu, ne_ni = plasma.ne_ni, use_p = true, rat_low = 1.0,
                      rat_high = 10.0, gamma = 13.0 / 9.0, thermal_frac = plasma.thermal_frac,
                      power_frac = plasma.power_frac, kappa_frac = plasma.kappa_frac,
                      power_p = plasma.power_p, gamma_min = plasma.gamma_min,
                      gamma_max = plasma.gamma_max, kappa = plasma.kappa, w = plasma.w,
                      sigma_cut = plasma.sigma_cut, theta_e_min = -1.0)
    return RunConfig(m_bh_cgs = BL_MBH, mass_scale_cgs = ms, frequency_hz = nu,
                     camera = CameraConfig(inclination_deg = inc, distance = dcam, fov = fov,
                                           resolution = res),
                     plasma = pl, max_steps = max_steps, C = C, escape_factor = 1.1)
end

"pixel solid angle (sr) of a camera of full width `fov` M and `res` pixels per side"
pixel_sr(fov, res) = (fov / res * BL_L / BL_DIST)^2

"flux density (Jy) of an intensity image (cgs) of the given camera"
flux_jy(I::AbstractMatrix, fov, res) = sum(I) * pixel_sr(fov, res) * 1.0e23

# JSON writer (strings, numbers, bools, vectors, dicts)
json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::Nothing) = "null"
json_str(x::Symbol) = json_str(string(x))
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(x::Tuple) = json_str(collect(x))
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = x -> string(first(x)))), ",\n ") * "}"
write_json(path, d) = open(io -> println(io, json_str(d)), path, "w")

# The record written beside every figure's data.
function record(name; kw...)
    d = Dict{String,Any}("figure" => name, "code" => BL_GIT_SHA, "date" => string(now()),
                         "threads" => Threads.nthreads(), "spin" => BL_SPIN, "m_bh_msun" => BL_MBH / MSUN,
                         "distance_kpc" => BL_DIST / 3.0857e21, "dump" => dump_path())
    for (k, v) in kw
        d[string(k)] = v
    end
    return d
end

"""
    sampled_path(plan, states, dls) -> NamedTuple

For one stored trajectory (states source → camera, `dls[n]` the affine step from state n
to n + 1), the per-state coordinate time `t`, position `x`, the Kerr-Schild proper length
increment `ds` of the segment to the next state, `dlam` the affine step, and for each
state the decoded plasma: `valid`, `n_e`, `theta_e`, `B`, `sigma`, `beta_inv`, `rho_cgs`
(from the primitives), with `j` and `alpha` the invariant Stokes-I coefficients for the
plan's parameters and the photon wave covector k = -p. The proper length increment is
sqrt(g_ij dx^i dx^j) of the straight segment, the time increment |dt|.
"""
function sampled_path(plan, states, dls)
    st = plan.spacetime
    pr = plan.provider
    pa = plan.params
    n = length(states)
    t = Vector{Float64}(undef, n); X = Vector{SVector{3,Float64}}(undef, n)
    ds = zeros(n); dt = zeros(n); dlam = zeros(n)
    valid = falses(n); n_e = zeros(n); th = zeros(n); B = zeros(n); sig = zeros(n); bi = zeros(n)
    rho = zeros(n); j = zeros(n); al = zeros(n)
    units = pa.units
    for k in 1:n
        y = states[k]
        x = SVector(y[1], y[2], y[3], y[4])
        t[k] = y[1]; X[k] = SVector(y[2], y[3], y[4])
        if k < n
            y2 = states[k+1]
            d = SVector(y2[2] - y[2], y2[3] - y[3], y2[4] - y[4])
            gcov = metric_and_inverse(st, x)[1]
            gs = SMatrix{3,3}(gcov[2,2], gcov[3,2], gcov[4,2], gcov[2,3], gcov[3,3], gcov[4,3], gcov[2,4], gcov[3,4], gcov[4,4])
            q = dot(d, gs * d)
            ds[k] = sqrt(max(q, 0.0))
            dt[k] = abs(y2[1] - y[1])
            dlam[k] = dls[k]
        end
        prim = primitives(pr, x)
        if prim.inside && prim.rho > 0.0
            state, g = sample(pr, st, x, pa)
            valid[k] = state.valid
            if state.valid
                n_e[k] = state.n_e; th[k] = state.theta_e; B[k] = state.B
                sig[k] = state.sigma; bi[k] = state.beta_inv
                rho[k] = prim.rho * units.dens_unit
                kcov = -SVector(y[5], y[6], y[7], y[8])
                jj, aa = coefficients(StokesI(), pa, state, kcov, pa.nu_obs)
                j[k] = jj; al[k] = aa
            else
                rho[k] = prim.rho * units.dens_unit
                sig[k] = NaN; bi[k] = NaN
            end
        end
    end
    return (t = t, x = X, ds = ds, dt = dt, dlam = dlam, valid = valid, n_e = n_e, theta_e = th,
            B = B, sigma = sig, beta_inv = bi, rho = rho, j = j, alpha = al)
end
