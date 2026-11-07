# fig_cmp_disk_krang.jl — Krang.jl side of the analytic thin-disk code comparison.
#
# Krang (observer at infinity, Bardeen screen) renders the AART disk as a custom material
# on Krang's equatorial cone geometry ConeGeometry(π/2): per screen point and order n,
# Krang's raytrace loops over its direct and indirect geodesic branches, finds the
# emission radius with emission_radius, and the material returns profile(r_s) g³ with
# g = 1/(−k_μ u^μ) from Krang's photon covector p_bl_d (k_t = −1) and the disk
# four-velocity of fig_cmp_disk_common.jl. The screen points are the grid nodes and the
# radial cuts written by fig_cmp_disk.jl (run that first).
#
# Writes figures/paper/out/cmp_disk/<config>/:
#   krang_I{n}.npy, krang_r{n}.npy     grid, [iβ, iα]; r NaN where Krang finds no order-n source
#   cut_krang_I{n}.npy, cut_krang_r{n}.npy
#   krang_meta.txt
# Env: BLP_CMP_CONFIGS (default "A,B").
# Run from the repository root:
#   julia -t 16 --project=results/blp/rt_compare/disk/krang_env figures/paper/fig_cmp_disk_krang.jl
# (an environment with Krang developed from its checkout and BlackLightPlus from this one).

using Krang
include(joinpath(@__DIR__, "fig_cmp_disk_common.jl"))
using BlackLightPlus.Matter: johnson_su_profile

"The AART disk as a Krang material: observed intensity of the orders `subimgs`."
struct AARTDiskMaterial{N} <: Krang.AbstractMaterial
    subimgs::NTuple{N,Int}
    disk::EquatorialDisk{Float64}
end
Krang.isFastLight(::AARTDiskMaterial) = true
Krang.isAxisymmetric(::AARTDiskMaterial) = true

function (m::AARTDiskMaterial)(pix::Krang.AbstractPixel, intersection)
    (; rs, θs, νr, νθ) = intersection
    met = Krang.metric(pix)
    α, β = Krang.screen_coordinate(pix)
    θo = Krang.inclination(pix)
    η = Krang.η(met, α, β, θo)
    λ = Krang.λ(met, α, θo)
    k = Krang.p_bl_d(met, rs, θs, η, λ, νr, νθ)          # (k_t, k_r, k_θ, k_φ), k_t = −1
    ut, ur, uφ = disk_velocity(rs, m.disk)
    g = 1 / (-(k[1] * ut + k[2] * ur + k[4] * uφ))
    d = m.disk
    return johnson_su_profile(rs; mu = d.mu_p, sigma = d.sigma_p, gamma = d.gamma_p) *
           g^d.redshift_power
end

"Krang's order-n emission radius at `pix` (NaN when neither branch has a source)."
function krang_radius(pix, n)
    r = NaN
    for isindir in (true, false)
        rs, _, _, _, ok = Krang.emission_radius(pix, π / 2, isindir, n)
        (ok && Krang.horizon(Krang.metric(pix)) < rs < Inf) && (r = rs)
    end
    return r
end

function krang_points(met, θo, αs, βs, disk)
    np = length(αs)
    I = [zeros(np) for _ in 1:3]; R = [fill(NaN, np) for _ in 1:3]
    meshes = ntuple(n -> Krang.Mesh(Krang.ConeGeometry(π / 2), AARTDiskMaterial((n - 1,), disk)), 3)
    Threads.@threads :dynamic for i in 1:np
        pix = Krang.IntensityPixel(met, αs[i], βs[i], θo)
        for n in 1:3
            I[n][i] = Krang.raytrace(pix, meshes[n])
            R[n][i] = krang_radius(pix, n - 1)
        end
    end
    return I, R
end

function load_vec(path)
    open(path) do io
        read(io, 8); hl = Int(read(io, UInt16)); read(io, hl)
        return reinterpret(Float64, read(io))[:]
    end
end

function run_krang(cfg)
    dir = cmp_dir(cfg)
    met = Krang.Kerr(cfg.a); θo = deg2rad(cfg.inc)
    disk = EquatorialDisk(cfg.a)
    camera = CameraConfig(inclination_deg = cfg.inc, distance = cfg.D, fov = cmp_fov(cfg),
                          resolution = cfg.N)
    centres = prepare(RunConfig(camera = camera); spacetime = KerrSchildCartesian(cfg.a),
                      matter = nothing).pixels.centres
    N = cfg.N
    gα = vec([centres[j] for i in 1:N, j in 1:N])          # column-major [iβ, iα]
    gβ = vec([centres[i] for i in 1:N, j in 1:N])
    t_g = @elapsed I, R = krang_points(met, θo, gα, gβ, disk)
    for n in 0:2
        save_npy(joinpath(dir, "krang_I$n.npy"), reshape(I[n + 1], N, N))
        save_npy(joinpath(dir, "krang_r$n.npy"), reshape(R[n + 1], N, N))
    end
    cα = load_vec(joinpath(dir, "cut_alpha.npy")); cβ = load_vec(joinpath(dir, "cut_beta.npy"))
    t_c = @elapsed cI, cR = krang_points(met, θo, cα, cβ, disk)
    for n in 0:2
        save_npy(joinpath(dir, "cut_krang_I$n.npy"), cI[n + 1])
        save_npy(joinpath(dir, "cut_krang_r$n.npy"), cR[n + 1])
    end
    open(joinpath(dir, "krang_meta.txt"), "w") do io
        println(io, "git=", cmp_git_sha(), " krang=", pkgversion(Krang), " at ", pkgdir(Krang))
        @printf(io, "config=%s a=%.17g inc=%.17g N=%d observer=infinity\n", cfg.name, cfg.a, cfg.inc, N)
        @printf(io, "run: threads=%d grid=%.1f s cuts=%.1f s (%d points)\n", Threads.nthreads(), t_g, t_c, length(cα))
    end
    @printf("[%s] Krang grid %d^2 %.1f s, cuts %d points %.1f s\n", cfg.name, N, t_g, length(cα), t_c)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for name in split(get(ENV, "BLP_CMP_CONFIGS", "A,B"), ",")
        run_krang(cmp_config(strip(name)))
    end
end
