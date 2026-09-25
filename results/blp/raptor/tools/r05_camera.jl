# r05_camera.jl — R05 camera check (dag.md R05): RAPTOR's initial ray states from its own
# camera (init_camera blocks + initialize_photon with CAMERA_PP, M4) against BlackLightPlus
# `camera_rays` for 480^2 pixels at i = 17, 60, 90 deg (s455 binary, d = 1000, FOV 56).
#
# Usage: julia --project=. r05_camera.jl <libraptor_sks_fd.so> <out.json>
# Compares x^mu (max |dx| / max(|x|, 1)) and k_mu (RAPTOR k^mu lowered with the
# BlackLightPlus metric against k_mu = -p_mu); exit 0 iff both are <= 1e-14.

using BlackLightPlus, StaticArrays, LinearAlgebra, Printf
using BlackLightPlus.Geodesics: camera_rays, PlaneParallelCamera, metric, inverse_metric
include(joinpath(@__DIR__, "r04_legs.jl"))
include(joinpath(@__DIR__, "jsonw.jl"))

lib, outjson = ARGS[1], ARGS[2]
const NPIX = 480
const FOV = 56.0
const DIST = 1000.0
h = Libc.Libdl.dlopen(lib)
fload = Libc.Libdl.dlsym(h, :raptor_load_bbh)
fcam = Libc.Libdl.dlsym(h, :raptor_camera_image)
bbh = write_bbh_in(joinpath(mktempdir(), "s455.in"), "s455"; cam_dist = DIST)
ccall(fload, Cvoid, (Cstring,), bbh)
st = leg_spacetime("s455")

res = Dict{String,Any}(); pass = true
for inc in (17.0, 60.0, 90.0)
    rays = camera_rays(st, PlaneParallelCamera(inc_deg = inc, d = DIST, fov = FOV, npix = NPIX))
    buf = zeros(8 * NPIX * NPIX)
    ccall(fcam, Cvoid, (Cint, Cdouble, Cdouble, Ptr{Float64}), NPIX, FOV, inc, buf)
    dx = 0.0; dkd = 0.0; dku = 0.0
    for i in eachindex(rays)
        y = rays[i]; r = @view buf[8(i-1)+1:8i]
        x = SVector(y[1], y[2], y[3], y[4])
        for m in 1:4
            dx = max(dx, abs(r[m] - x[m]) / max(abs(x[m]), 1.0))
        end
        kd_blp = -SVector(y[5], y[6], y[7], y[8])
        ku_r = SVector(r[5], r[6], r[7], r[8])
        dkd = max(dkd, maximum(abs.(metric(st, x) * ku_r - kd_blp)))
        dku = max(dku, maximum(abs.(inverse_metric(st, x) * kd_blp - ku_r)))
    end
    ok = dx <= 1e-14 && dkd <= 1e-14
    global pass &= ok
    res["i$(Int(inc))"] = Dict("max_rel_dx" => dx, "max_abs_dk_cov" => dkd,
                               "max_abs_dk_con" => dku, "npix" => NPIX, "pass" => ok)
    @printf("i=%2d  x: %.2e  k_mu: %.2e  k^mu: %.2e  %s\n", inc, dx, dkd, dku, ok ? "PASS" : "FAIL")
end
res["pass"] = pass
res["leg"] = "s455 (d = 1000, FOV 56)"
mkpath(dirname(outjson))
write_json(outjson, res)
exit(pass ? 0 : 1)
