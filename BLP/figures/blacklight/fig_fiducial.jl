# fig_fiducial.jl — the fiducial-deck images of paper figures 4, 5, 6 and 10:
#   stokes    — Stokes I, Q, U, V of the fiducial image (figure 4);
#   cuts      — the fiducial image (sigma <= 1), the image of the beta^-1 >= 2 plasma only,
#               and the images of the near and far half-spaces about the plane through
#               the origin orthogonal to the line of sight (figure 5), the cuts applied to
#               the primitives through a masked provider;
#   flat      — the fiducial image beside the image traced in flat spacetime (figure 6);
#   nonthermal— thermal electrons, a power law (p = 3, gamma 4 .. 1000), a kappa
#               distribution (kappa = 4, w = 1/2), and the equal three-way mix (figure 10).
# Every image: the paper camera (r = 100 M, theta = 45 deg, width 24 M, 256^2), 230 GHz,
# the calibrated mass scale. Writes out/blacklight/<set>/*.npy and record.json.
# Run from the repository root:   julia -t 32 --project=BLP BLP/figures/blacklight/fig_fiducial.jl
include(joinpath(@__DIR__, "common.jl"))

# A grid provider that keeps the primitives of `p` where `keep(x, prim)` is true.
struct MaskedProvider{P,F} <: GridProvider
    p::P
    keep::F
end
BLP.Matter.primitives(m::MaskedProvider, x::SVector{4}) = begin
    prim = primitives(m.p, x)
    (prim.inside && m.keep(x, prim)) ? prim : PlasmaPrimitives{typeof(prim.rho)}()
end
BLP.Matter.domain_radius(m::MaskedProvider) = domain_radius(m.p)

function save_image(dir, tag, img)
    for (n, A) in (("I", img.I), ("Q", img.Q), ("U", img.U), ("V", img.V))
        write_npy(joinpath(dir, "$(tag)_$n.npy"), A)
    end
    return flux_jy(img.I, BL_FOV, BL_RES)
end

function main()
    snap = AthenaKSnapshot(dump_path())
    kerr = KerrSchildCartesian(BL_SPIN)
    cfg = fiducial_config()
    rec = Dict{String,Any}()

    # figure 4: Stokes parameters
    d = outdir("stokes")
    t0 = time()
    img = render(prepare(cfg; spacetime = kerr, matter = snap, mode = IQUV()); backend = :cpu)
    rec["stokes"] = Dict("flux_jy" => save_image(d, "fiducial", img), "seconds" => time() - t0,
                         "lp" => sum(sqrt.(img.Q .^ 2 .+ img.U .^ 2)) / sum(img.I), "cp" => sum(img.V) / sum(img.I))
    logf("stokes: %.3f Jy in %.0f s\n", rec["stokes"]["flux_jy"], rec["stokes"]["seconds"])
    write_json(joinpath(d, "record.json"), record("stokes"; result = rec["stokes"]))

    # figure 5: cuts
    d = outdir("cuts")
    si, ci = sincosd(BL_INC)
    dhat = SVector(si, 0.0, ci)
    γ = cfg.plasma.gamma
    beta_keep = (x, prim) -> begin
        gcov, gcon = metric_and_inverse(kerr, x)
        b_sq = primitive_four_vectors(prim.uu, prim.bb, gcov, gcon)[5]
        b_sq / (2 * pgas_from_eint(prim.eint, γ)) >= 2.0
    end
    near_keep = (x, prim) -> (x[2] * dhat[1] + x[3] * dhat[2] + x[4] * dhat[3]) > 0.0
    far_keep = (x, prim) -> (x[2] * dhat[1] + x[3] * dhat[2] + x[4] * dhat[3]) < 0.0
    cuts = Dict{String,Any}()
    for (tag, prov) in (("fiducial", snap), ("beta_inv_ge_2", MaskedProvider(snap, beta_keep)),
                        ("near_half", MaskedProvider(snap, near_keep)), ("far_half", MaskedProvider(snap, far_keep)))
        t0 = time()
        img = render(prepare(cfg; spacetime = kerr, matter = prov, mode = StokesI()); backend = :cpu)
        cuts[tag] = Dict("flux_jy" => save_image(d, tag, img), "seconds" => time() - t0)
        logf("cuts %s: %.3f Jy in %.0f s\n", tag, cuts[tag]["flux_jy"], cuts[tag]["seconds"])
    end
    write_json(joinpath(d, "record.json"), record("cuts"; cuts = cuts, plane_normal = collect(dhat)))

    # figure 6: flat spacetime
    d = outdir("flat")
    flat = Dict{String,Any}()
    for (tag, stc) in (("kerr", kerr), ("flat", Minkowski()))
        t0 = time()
        img = render(prepare(cfg; spacetime = stc, matter = snap, mode = StokesI()); backend = :cpu)
        flat[tag] = Dict("flux_jy" => save_image(d, tag, img), "seconds" => time() - t0)
        logf("flat %s: %.3f Jy in %.0f s\n", tag, flat[tag]["flux_jy"], flat[tag]["seconds"])
    end
    write_json(joinpath(d, "record.json"), record("flat"; runs = flat))

    # figure 10: electron distributions
    d = outdir("nonthermal")
    nt = Dict{String,Any}()
    decks = (("thermal", PlasmaConfig()),
             ("powerlaw", PlasmaConfig(thermal_frac = 0.0, power_frac = 1.0, power_p = 3.0, gamma_min = 4.0, gamma_max = 1000.0)),
             ("kappa", PlasmaConfig(thermal_frac = 0.0, kappa_frac = 1.0, kappa = 4.0, w = 0.5)),
             ("mixed", PlasmaConfig(thermal_frac = 1 / 3, power_frac = 1 / 3, kappa_frac = 1 / 3, power_p = 3.0,
                                    gamma_min = 4.0, gamma_max = 1000.0, kappa = 4.0, w = 0.5)))
    for (tag, pl) in decks
        t0 = time()
        c = fiducial_config(; plasma = pl)
        img = render(prepare(c; spacetime = kerr, matter = snap, mode = StokesI()); backend = :cpu)
        nt[tag] = Dict("flux_jy" => save_image(d, tag, img), "seconds" => time() - t0)
        logf("nonthermal %s: %.3f Jy in %.0f s\n", tag, nt[tag]["flux_jy"], nt[tag]["seconds"])
    end
    write_json(joinpath(d, "record.json"), record("nonthermal"; runs = nt))
end
main()
