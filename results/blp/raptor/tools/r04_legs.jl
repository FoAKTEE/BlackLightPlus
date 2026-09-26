# r04_legs.jl — the R04/R05 comparison legs (dag.md R04): binaries as BlackLightPlus
# `SuperposedKerrSchild` and as RAPTOR bbh.in files.
#   kerr : one hole, m = 1, a = 0.9375, at the origin (SKS with N = 1)
#   s455 : m = 0.5 + 0.5, a = 0, C1 = -C2 = (12.19120680, -2.76124551, 0)   (design.md §2)
#   s122 : m = 0.5 + 0.5, a = 0.4 along +z, C1 = -C2 = (0.32159304, 7.99353351, 0)
#   s093 : m = 0.5 + 0.5, a = 0.4 along +z, C1 = -C2 = (-10.42584453, 6.89577884, 0)  (R09)
#   s122r: s122 with both spins along -z (a = -0.4; R09 spin-orientation leg)

using BlackLightPlus, StaticArrays
using BlackLightPlus.Geodesics: metric, inverse_metric, christoffel, ks_radius, KerrSchildCartesian
const GEO = BlackLightPlus.Geodesics

const LEGS = Dict(
    "kerr" => (m = [1.0], a = [0.9375], c = [(0.0, 0.0, 0.0)]),
    "s455" => (m = [0.5, 0.5], a = [0.0, 0.0],
               c = [(12.19120680, -2.76124551, 0.0), (-12.19120680, 2.76124551, 0.0)]),
    "s122" => (m = [0.5, 0.5], a = [0.4, 0.4],
               c = [(0.32159304, 7.99353351, 0.0), (-0.32159304, -7.99353351, 0.0)]),
    "s093" => (m = [0.5, 0.5], a = [0.4, 0.4],
               c = [(-10.42584453, 6.89577884, 0.0), (10.42584453, -6.89577884, 0.0)]),
    "s122r" => (m = [0.5, 0.5], a = [-0.4, -0.4],
                c = [(0.32159304, 7.99353351, 0.0), (-0.32159304, -7.99353351, 0.0)]),
)

leg_spacetime(leg) = (L = LEGS[leg]; SuperposedKerrSchild(L.m, L.a, L.c))

"Write the RAPTOR bbh.in of `leg` with march control C, budget nmax, escape radius and camera distance."
function write_bbh_in(path, leg; C = 80.0, nmax = 4000, escape = 1100.0, cam_dist = 1000.0,
                      horizon_eps = 1e-3, capture_redshift = 100.0)
    L = LEGS[leg]
    open(path, "w") do io
        println(io, "# R04 leg ", leg)
        println(io, "nholes ", length(L.m))
        println(io, "mass ", join(repr.(L.m), " "))
        println(io, "spin ", join(repr.(L.a), " "))
        for (A, c) in enumerate(L.c)
            println(io, "center$A ", join(repr.(c), " "))
        end
        println(io, "C ", repr(C))
        println(io, "nmax ", nmax)
        println(io, "escape ", repr(escape))
        println(io, "horizon_eps ", repr(horizon_eps))
        println(io, "capture_redshift ", repr(capture_redshift))
        println(io, "cam_dist ", repr(cam_dist))
    end
    return path
end

"Minimum over holes of r_A - r_h,A (BlackLightPlus)."
function horizon_gap(st, x::SVector{4})
    return minimum(GEO.ks_radius_hole(st, x, A) - GEO.horizon_radius_hole(st, A)
                   for A in eachindex(st.mass))
end
