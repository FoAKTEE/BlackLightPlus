# BlackLightPlus images of the comparison deck with the Faraday terms zeroed: full, norho (ρ_Q = ρ_V = 0),
# norhoq (ρ_Q = 0), norhov (ρ_V = 0), at 230 GHz and 1 THz. The thermal fit is redefined inside the
# Transfer module from its own source text with the selected rotativities zeroed.
using Printf
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN
const T = BlackLightPlus.Transfer
const ROOT = "/data/haiyangw/claude/BlackLightPlus"
const OUT = joinpath(ROOT, "figures", "out", "kpolaris", "diag_v")
const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"

src = read(joinpath(ROOT, "src", "physics", "Transfer", "synchrotron.jl"), String)
i0 = findfirst("@inline function thermal_synchrotron_polarized", src).start
i1 = findnext("return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)\nend", src, i0)
body = src[i0:i1.stop]
function redefine(zq::Bool, zv::Bool)
    ret = "return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, $(zq ? "0.0 * rho_q" : "rho_q"), $(zv ? "0.0 * rho_v" : "rho_v"))\nend"
    Base.eval(T, Meta.parse(replace(body, "return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)\nend" => ret)))
end
function config(nu)
    m_bh = 6.2e9 * MSUN; L = GNEWT * m_bh / CL^2
    RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = nu,
              camera = CameraConfig(inclination_deg = 60.0, distance = 100.0, fov = 30.0, resolution = 256),
              plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, use_p = true,
                                    sigma_cut = 1.0, theta_e_min = -1.0),
              max_steps = 2500, C = 80.0, escape_factor = 1.1)
end
snap = AthenaKSnapshot(DATA)
for (variant, zq, zv) in (("full", false, false), ("norho", true, true), ("norhoq", true, false), ("norhov", false, true))
    redefine(zq, zv)
    for (tag, nu) in (("1000", 1.0e12), ("230", 2.3e11))
        plan = prepare(config(nu); spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
        img = render(plan; backend = :cpu)
        for (n, A) in (("I", img.I), ("Q", img.Q), ("U", img.U), ("V", img.V))
            write_npy(joinpath(OUT, "blp_$(variant)_$(tag)_$n.npy"), A)
        end
        @printf("%-7s %5s GHz  sum I %.4e  V/I %+.5f  max|V|/maxI %.4f  %.1f s\n", variant, tag, sum(img.I),
                sum(img.V) / sum(img.I), maximum(abs.(img.V)) / maximum(img.I), img.run.seconds)
    end
end
println("blp variants done")
