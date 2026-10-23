# fig_selected_ray.jl — selected-ray analysis (KPolaris figure 4) on the torus dump:
# the full-Stokes image at 86 GHz and i = 85° (fiducial plasma deck of fig_pol_hires.jl,
# camera at d = 1000 M, fov 40 M, 256²), one selected pixel (the largest |V| among pixels
# brighter than 0.3 max I), and the polarized transfer along its ray recorded sample by
# sample: the march of `BlackLightPlus.Transfer.march(::IQUV)` (fixed control C = 80, the camera frame
# parallel-transported with the geodesic) is replayed here with every segment operator
# stored in the transported camera frame, then composed source → camera to give the Stokes
# vector after each segment. The displayed outgoing segment starts at the first sample
# after which the invariant intensity stays within 1% of its final value; the replay
# without conversion sets ρ_Q = 0 on that segment from the same incident Stokes vector.
# The intrinsic circular emission and Faraday conversion intervals are the 5%–95% span of
# the cumulative absolute stepwise contributions of j_V ds and ρ_Q U ds (fluid frame),
# each weighted by the subsequent scalar transmission exp(−τ_I), in forward path order.
# Writes figures/paper/out/kpolaris/selected_ray/: image_{I,Q,U,V}.npy ([iβ, iα]), path.npy
# (rows per sample camera → source: t, x, y, z, I, Q, U, V of the forward solution,
# transmission to the camera, j_V ds, ρ_Q U ds, valid), replay.npy (rows per sample of
# the segment: I, Q, U, V full, I, Q, U, V without conversion), slice_rho.npy (ρ in the
# z = 0 plane, ±25 M, 200²), record.json.
# Run: julia -t 32 --project=. figures/kpolaris/fig_selected_ray.jl
using Printf, Dates, LinearAlgebra, StaticArrays
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN
using BlackLightPlus.Spacetime: camera_ray, metric_and_inverse, geodesic_transport_rhs, march_control,
                     MarchBudget
using BlackLightPlus.Matter: sample, primitives
using BlackLightPlus.Transfer: SpacetimeMarch, coefficients, fluid_tetrad, fluid_tetrad_checked,
                    normal_observer, qu_rotation, rotate_qu, segment, stokes_admissible, IQUV,
                    max_steps, geodesic_step, next_state, renorm_e1

const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"
const OUT = joinpath(@__DIR__, "..", "out", "kpolaris", "selected_ray")
const RES = 256
const FOV = 40.0
const INC = 85.0
const NU = 86.0e9
mkpath(OUT)

m_bh = 6.2e9 * MSUN
L = GNEWT * m_bh / CL^2
cfg = RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = NU,
                camera = CameraConfig(inclination_deg = INC, distance = 1000.0, fov = FOV,
                                      resolution = RES),
                plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                      use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
                max_steps = 2500, C = 80.0, escape_factor = 1.1)
snap = AthenaKSnapshot(DATA)
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
img = render(plan; backend = :cpu)
for (n, A) in (("I", img.I), ("Q", img.Q), ("U", img.U), ("V", img.V))
    write_npy(joinpath(OUT, "image_$n.npy"), A)
end
@printf("image: sum I %.4e  %.1f s\n", sum(img.I), img.run.seconds)

# the selected pixel: largest |V| among pixels brighter than 0.3 max I
centres = range(-FOV / 2 + FOV / (2RES), FOV / 2 - FOV / (2RES); length = RES)
mask = img.I .> 0.3 * maximum(img.I)
score = abs.(img.V) .* mask
ib, ia = Tuple(argmax(score))                     # [iβ, iα]
α_sel, β_sel = centres[ia], centres[ib]
@printf("selected pixel (α, β) = (%.3f, %.3f): I %.3e  V/I %.4f  LP %.3f\n", α_sel, β_sel,
        img.I[ib, ia], img.V[ib, ia] / img.I[ib, ia],
        hypot(img.Q[ib, ia], img.U[ib, ia]) / img.I[ib, ia])

# the recorded march of this ray (the operations of Transfer.march(::IQUV))
st = plan.spacetime
p = plan.params
prov = plan.provider
ctl = SpacetimeMarch(st, plan.budget)
renorm = renorm_e1(ctl)
y0, ok = camera_ray(st, plan.frame, α_sel, β_sel)
ok || error("no camera ray")
cu = SVector{4,Float64}(plan.frame.up)
x0 = SVector(y0[1], y0[2], y0[3], y0[4]); kcov0 = -SVector(y0[5], y0[6], y0[7], y0[8])
gcov0, gcon0 = metric_and_inverse(st, x0); kcon0 = gcon0 * kcov0
e0c = normal_observer(gcon0); e0cov0 = gcov0 * e0c
upc0 = e0c[1] * cu[1] - (e0cov0[2] * cu[2] + e0cov0[3] * cu[3] + e0cov0[4] * cu[4]) / e0cov0[1]
upcam = SVector(upc0, cu[2] + e0c[2] * cu[1], cu[3] + e0c[3] * cu[1], cu[4] + e0c[4] * cu[1])
Ecam = fluid_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0; renorm_e1 = renorm)
eU = SVector(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4]); eUu = e0c

Ms = SMatrix{4,4,Float64}[]; bs = SVector{4,Float64}[]            # camera-frame operators
Mnc = SMatrix{4,4,Float64}[]; bnc = SVector{4,Float64}[]          # the same without ρ_Q
pos = SVector{4,Float64}[]; jV = Float64[]; rQ = Float64[]; dss = Float64[]
rot = Tuple{Float64,Float64}[]; alI = Float64[]; valid = Bool[]
y = y0; r_prev = Inf
for it in 1:max_steps(ctl)
    global y, r_prev, eU, eUu
    m = march_control(ctl, y, r_prev, it)
    m.stop && break
    h = m.h; r_prev = m.R
    x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
    state, g = sample(prov, st, x, p)
    push!(pos, x)
    M = one(SMatrix{4,4,Float64}); b = zero(SVector{4,Float64}); M2 = M; b2 = b
    jv = 0.0; rq = 0.0; ds = 0.0; c2 = 1.0; s2 = 0.0; ai = 0.0; ok_s = false
    if state.valid
        gcov, gcon = g
        kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8]); kcon = gcon * kcov
        j, al, ro, tet_fl, vld = coefficients(IQUV(), p, state, kcov, kcon, g, p.nu_obs, renorm)
        if vld
            ucov = gcov * eUu
            E, screen_ok = fluid_tetrad_checked(eUu, ucov, kcon, kcov, eU, gcov, gcon; renorm_e1 = renorm)
            screen_ok || break
            ds = h * p.units.L_unit / p.nu_obs
            Mf, bf = segment(IQUV(), j, al, ro, ds)
            Mf2, bf2 = segment(IQUV(), j, al, SVector(0.0, 0.0, 0.0, ro[4]), ds)
            c2, s2 = qu_rotation(tet_fl, E, gcov)
            M = rotate_qu(c2, s2, Mf); b = rotate_qu(c2, s2, bf)
            M2 = rotate_qu(c2, s2, Mf2); b2 = rotate_qu(c2, s2, bf2)
            jv = j[4]; rq = ro[2]; ai = al[1]; ok_s = true
        end
    end
    push!(Ms, M); push!(bs, b); push!(Mnc, M2); push!(bnc, b2); push!(jV, jv); push!(rQ, rq)
    push!(dss, ds); push!(rot, (c2, s2)); push!(alI, ai); push!(valid, ok_s)
    dy, (de, du) = geodesic_transport_rhs(st, y, (eU, eUu))
    ay = dy; ae = de; au = du
    ys = y + (h / 2) * dy; es = eU + (h / 2) * de; us = eUu + (h / 2) * du
    dy, (de, du) = geodesic_transport_rhs(st, ys, (es, us)); ay += 2dy; ae += 2de; au += 2du
    ys = y + (h / 2) * dy; es = eU + (h / 2) * de; us = eUu + (h / 2) * du
    dy, (de, du) = geodesic_transport_rhs(st, ys, (es, us)); ay += 2dy; ae += 2de; au += 2du
    ys = y + h * dy; es = eU + h * de; us = eUu + h * du
    dy, (de, du) = geodesic_transport_rhs(st, ys, (es, us)); ay += dy; ae += de; au += du
    y = next_state(ctl, y + (h / 6) * ay, it); eU += (h / 6) * ae; eUu += (h / 6) * au
end
n = length(Ms)
# forward composition source → camera: S_k = stokes vector after segment k (camera frame)
S = zeros(SVector{4,Float64}, n)
acc = zero(SVector{4,Float64})
for k in n:-1:1
    global acc
    acc = stokes_admissible(Ms[k] * acc + bs[k])
    S[k] = acc
end
Scam = S[1] * p.nu_obs^3
@printf("replayed ray: %d samples, %d emitting; Stokes at camera %s vs image pixel %s\n", n,
        count(valid), string(round.(Scam; sigdigits = 5)),
        string(round.((img.I[ib, ia], img.Q[ib, ia], img.U[ib, ia], img.V[ib, ia]); sigdigits = 5)))
# transmission from sample k to the camera: Π_{m<k} e^{-α_I ds}
trans = ones(n)
for k in 2:n
    trans[k] = trans[k-1] * exp(-alI[k-1] * dss[k-1])
end
# contributions in forward order (k from n down to 1): emission j_V ds, conversion ρ_Q U_fluid ds
function Ufluid(k)                                               # U in the fluid frame (R⁻¹ S)
    c, s = rot[k]
    return c * S[k][3] - s * S[k][2]
end
cem = zeros(n); ccv = zeros(n)
for k in n:-1:1
    cem[k] = abs(jV[k] * dss[k]) * trans[k]
    ccv[k] = abs(rQ[k] * Ufluid(k) * dss[k]) * trans[k]
end
function interval(c)
    tot = sum(c); tot > 0 || return (0, 0)
    cum = 0.0; k5 = 0; k95 = 0
    for k in n:-1:1
        cum += c[k]
        k5 == 0 && cum >= 0.05tot && (k5 = k)
        k95 == 0 && cum >= 0.95tot && (k95 = k; break)
    end
    return (k5, k95)                                   # sample indices (camera-ordered)
end
em_iv = interval(cem); cv_iv = interval(ccv)
# the displayed outgoing segment: first sample (forward order) after which I stays within 1%
Ifin = S[1][1]
kseg = 1
for k in n:-1:1
    global kseg
    if all(abs(S[m][1] - Ifin) <= 0.01 * Ifin for m in k:-1:1)
        kseg = k; break
    end
end
# replay on the segment: full and without conversion, from the same incident vector
S0 = kseg < n ? S[kseg+1] : zero(SVector{4,Float64})
full = SVector{4,Float64}[]; noconv = SVector{4,Float64}[]
a1 = S0; a2 = S0
for k in kseg:-1:1
    global a1, a2
    a1 = stokes_admissible(Ms[k] * a1 + bs[k]); a2 = stokes_admissible(Mnc[k] * a2 + bnc[k])
    push!(full, a1); push!(noconv, a2)
end
pol(v) = sqrt(v[2]^2 + v[3]^2 + v[4]^2)
@printf("segment: samples %d..1 (%d), full: Π/I %.4f V/I %.4f; no conversion: Π/I %.4f V/I %.4f\n",
        kseg, kseg, pol(a1) / a1[1], a1[4] / a1[1], pol(a2) / a2[1], a2[4] / a2[1])
# the density slice in the orbital plane
nsl = 200; half = 25.0
slice = Matrix{Float64}(undef, nsl, nsl)
for (j, yy) in enumerate(range(-half, half; length = nsl)), (i, xx) in enumerate(range(-half, half; length = nsl))
    slice[j, i] = primitives(prov, SVector(0.0, xx, yy, 0.0)).rho
end
write_npy(joinpath(OUT, "slice_rho.npy"), slice)
P = Matrix{Float64}(undef, n, 12)
for k in 1:n
    P[k, 1:4] .= pos[k]; P[k, 5:8] .= S[k]; P[k, 9] = trans[k]; P[k, 10] = cem[k]
    P[k, 11] = ccv[k]; P[k, 12] = valid[k]
end
write_npy(joinpath(OUT, "path.npy"), P)
R = Matrix{Float64}(undef, length(full), 8)
for k in eachindex(full)
    R[k, 1:4] .= full[k]; R[k, 5:8] .= noconv[k]
end
write_npy(joinpath(OUT, "replay.npy"), R)
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::Integer) = string(x)
json(x::AbstractString) = "\"$x\""
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ", ") * "}"
rec = Dict("res" => RES, "fov" => FOV, "inc_deg" => INC, "nu_hz" => NU, "pixel_ab" => [α_sel, β_sel],
           "pixel_index_ba" => [ib, ia], "pixel_stokes" => [img.I[ib, ia], img.Q[ib, ia], img.U[ib, ia], img.V[ib, ia]],
           "replay_stokes" => collect(Scam), "nsamples" => n, "nemitting" => count(valid),
           "segment_start_sample" => kseg, "emission_interval" => collect(em_iv),
           "conversion_interval" => collect(cv_iv),
           "full_pi_over_i" => pol(a1) / a1[1], "full_v_over_i" => a1[4] / a1[1],
           "noconv_pi_over_i" => pol(a2) / a2[1], "noconv_v_over_i" => a2[4] / a2[1],
           "slice_half" => half, "slice_n" => nsl, "date" => string(now()),
           "threads" => Threads.nthreads())
open(io -> println(io, json(rec)), joinpath(OUT, "record.json"), "w")
println("wrote ", OUT)
