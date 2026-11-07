# fig_cmp_disk_common.jl — shared definitions of the analytic thin-disk code comparison
# (AART, Krang.jl, BlackLightPlus closed form, BlackLightPlus RK4 march).
#
# Included by fig_cmp_disk.jl (BlackLightPlus, both tracers) and fig_cmp_disk_krang.jl
# (Krang); the AART side reads the same configurations from configs.json, written here.
#
# The disk is AART's: Johnson-SU emissivity (γ = −3/2, μ = 1 − √(1 − a²), σ = 1/2), a
# Keplerian flow outside the ISCO that plunges with the ISCO energy and angular momentum
# inside it (sub_kep = β_r = β_φ = 1), observed intensity I = profile(r_s) g³, image orders
# n = 0, 1, 2. Every code samples the point values of I_n at the same screen points: the
# nodes of AART's Cartesian grid, linspace(−L, L, N) on each axis (equal to the pixel
# centres of a BlackLightPlus camera of N pixels and field of view 2L N/(N − 1)), and a set
# of radial cuts through the photon rings.
#
# Arrays are laid out [iβ, iα] (numpy axis 0 = β, axis 1 = α).

using Printf
using BlackLightPlus
using BlackLightPlus.Spacetime: kerr_delta, isco_prograde
using BlackLightPlus.Matter: l_subkep, omega_subkep, omega_infall, u_r_infall, u_t_general,
                             E_subkep, nu_subkep

"Output root; `BLP_CMP_OUT` redirects it (a scratch copy to diff against the kept arrays)."
const CMP_OUT = get(ENV, "BLP_CMP_OUT", joinpath(@__DIR__, "out", "cmp_disk"))

"""
Comparison configurations. `L`, `N`: grid nodes linspace(−L, L, N). Observer distances:
`D` of AART and the closed-form tracer, whose radial integrals start at r = D on the ray of
Bardeen coordinates (α, β), an offset O(1/D) from the observer at infinity of Krang (1/D
at D = 10¹⁵ is below rounding); `D_fin`, AART's default observer distance, a second
closed-form run that measures that offset; `D_rk`, the camera distance of the RK4 march
(plane-parallel rays, whose conserved quantities differ from Bardeen's by O(1/D_rk)).
"""
const CMP_CONFIGS = (
    A = (name = "A", a = 0.94, inc = 17.0, L = 25.0, N = 2500, D = 1.0e15, D_fin = 1.0e5, D_rk = 1.0e8),
    B = (name = "B", a = 0.99, inc = 80.0, L = 12.0, N = 1200, D = 1.0e15, D_fin = 1.0e5, D_rk = 1.0e8),
)

cmp_config(name::AbstractString) = getproperty(CMP_CONFIGS, Symbol(name))

"Grid nodes linspace(−L, L, N) of AART's force_lims grid."
cmp_nodes(cfg) = collect(range(-cfg.L, cfg.L; length = cfg.N))

"Full field of view of the BlackLightPlus camera whose pixel centres are the AART nodes."
cmp_fov(cfg) = 2 * cfg.L * cfg.N / (cfg.N - 1)

cmp_dir(cfg) = (d = joinpath(CMP_OUT, cfg.name); mkpath(d); d)

"""
    save_npy(path, a)

`write_npy` for a `Float64` matrix; a vector is written as a one-dimensional `.npy`
array (format version 1.0, little-endian `<f8`).
"""
save_npy(path::AbstractString, a::AbstractMatrix) = write_npy(path, Matrix{Float64}(a))
function save_npy(path::AbstractString, v::AbstractVector)
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': ($(length(v)),), }"
    pad = 64 - mod(6 + 2 + 2 + length(header) + 1, 64)
    header = header * repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93], "NUMPY", UInt8[0x01, 0x00])
        write(io, UInt16(length(header)))
        write(io, header)
        write(io, Vector{Float64}(v))
    end
    return path
end

"The one-dimensional `<f8` array of a version-1.0 `.npy` file written by `save_npy`."
function load_npy_vector(path::AbstractString)
    open(path) do io
        String(read(io, 6)) == "\x93NUMPY" || error("$path: not an .npy file")
        read(io, 2)                                   # format version 1.0
        hl = Int(ltoh(read(io, UInt16)))
        header = String(read(io, hl))
        occursin("'<f8'", header) && occursin("False", header) ||
            error("$path: expected a C-order <f8 array, header $header")
        v = Vector{Float64}(undef, (filesize(path) - 10 - hl) ÷ 8)
        read!(io, v)
        return v
    end
end

# ------------------------------------------------------------------ disk four-velocity
"""
    disk_velocity(r, disk) -> (u^t, u^r, u^φ)

Contravariant Boyer–Lindquist four-velocity of the AART disk flow at the equatorial radius
`r` (the flow of `gfactor_circular` outside the ISCO and `gfactor_plunge` inside it).
"""
function disk_velocity(r, disk)
    a = disk.a
    isco = isco_prograde(a)
    if r >= isco
        ℓ = l_subkep(r, a, disk.sub_kep)
        OH = omega_subkep(r, a, ℓ)
        OT = OH + (1 - disk.betaphi) * (omega_infall(r, a) - OH)
        ur = (1 - disk.betar) * u_r_infall(r, a)
    else
        ℓms = l_subkep(isco, a, disk.sub_kep)
        OH = omega_subkep(r, a, ℓms)
        OT = OH + (1 - disk.betaphi) * (omega_infall(r, a) - OH)
        Ems = E_subkep(isco, a, ℓms)
        urhat = -kerr_delta(r, a) / (r * r) * nu_subkep(r, a, ℓms, Ems) * Ems
        ur = urhat + (1 - disk.betar) * (u_r_infall(r, a) - urhat)
    end
    ut = u_t_general(r, a, ur, OT)
    return ut, ur, ut * OT
end

# ------------------------------------------------------------------ radial cuts
"""
    cmp_cuts(cfg, rcrit; nphi = 120) -> (phi, rho, alpha, beta, nrho)

Radial cuts through the photon rings: `nphi` angles φ_k = (k − 1/2) 2π/nphi (never on
the α axis, where β = 0), and on each the radii ρ of a nonuniform grid: spacing 10⁻² over
(0, 0.95 L], refined to 10⁻⁴ on [ρc − 0.6, ρc + 0.4] and 10⁻³ on (ρc + 0.4, ρc + 2],
with ρc = `rcrit(φ)` the critical-curve radius along that angle. `alpha`, `beta` are the flattened screen points, angle-major (`nrho[k]` radii
on angle k).
"""
function cmp_cuts(cfg, rcrit::Function; nphi::Int = 120)
    phi = [(k - 0.5) * 2π / nphi for k in 1:nphi]
    rho_all = Vector{Vector{Float64}}(undef, nphi)
    for (k, φ) in enumerate(phi)
        rc = rcrit(φ)
        lo, hi, top = rc - 0.6, rc + 0.4, rc + 2.0
        coarse = filter(r -> r < lo || r > top, collect(0.01:0.01:0.95 * cfg.L))
        medium = collect(range(hi, top; step = 1.0e-3))[2:end]
        fine = collect(range(lo, hi; step = 1.0e-4))
        rho_all[k] = sort!(unique!(vcat(coarse, fine, medium)))
    end
    nrho = length.(rho_all)
    alpha = Float64[]; beta = Float64[]; rho = Float64[]
    for (k, φ) in enumerate(phi)
        append!(rho, rho_all[k])
        append!(alpha, rho_all[k] .* cos(φ))
        append!(beta, rho_all[k] .* sin(φ))
    end
    return phi, rho, alpha, beta, nrho
end

"Radius along the screen angle φ of the closed loop (αc, βc) around the origin."
function loop_radius(αc, βc, φ)
    n = length(αc)
    best = NaN
    for i in 1:n
        j = i == n ? 1 : i + 1
        # ray from the origin along (cos φ, sin φ) against the segment (αc[i], βc[i])-(αc[j], βc[j])
        ex = αc[j] - αc[i]; ey = βc[j] - βc[i]
        den = cos(φ) * ey - sin(φ) * ex
        abs(den) < 1e-300 && continue
        t = (αc[i] * ey - βc[i] * ex) / den
        s = (αc[i] * sin(φ) - βc[i] * cos(φ)) / den
        if t > 0 && -1e-12 <= s <= 1 + 1e-12
            best = isnan(best) ? t : max(best, t)
        end
    end
    return best
end

# ------------------------------------------------------------------ records
"Write `configs.json` for the AART driver and the plotter."
function write_configs_json()
    mkpath(CMP_OUT)
    open(joinpath(CMP_OUT, "configs.json"), "w") do io
        println(io, "{")
        ks = collect(keys(CMP_CONFIGS))
        for (i, k) in enumerate(ks)
            c = CMP_CONFIGS[k]
            @printf(io, "  \"%s\": {\"a\": %.17g, \"inc\": %.17g, \"L\": %.17g, \"N\": %d, \"D\": %.17g, \"D_fin\": %.17g, \"D_rk\": %.17g}%s\n",
                    c.name, c.a, c.inc, c.L, c.N, c.D, c.D_fin, c.D_rk, i == length(ks) ? "" : ",")
        end
        println(io, "}")
    end
end

"Code version: HEAD SHA of this checkout, `-dirty` if src has uncommitted edits."
function cmp_git_sha()
    try
        root = dirname(dirname(@__DIR__))
        sha = readchomp(`git -C $root rev-parse HEAD`)
        isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
    catch
        "unknown"
    end
end
