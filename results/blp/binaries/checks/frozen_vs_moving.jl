include(joinpath(@__DIR__, "fvm_lib.jl"))
d = 300.0
function summ(name, as, f)
    lo, hi = edge(as, f)
    @printf("  %-34s captured %3d escaped %3d stuck %3d  interval [%.3f, %.3f] centre %.3f half-width %.3f\n", name, count(==(:captured), f), count(==(:escaped), f), count(==(:stuck), f), lo, hi, (lo+hi)/2, (hi-lo)/2)
end
O = SVector(0.0, 0.0, 0.0)

println("Calibration, v = 0 (static Schwarzschild), fine scan of the edge, exact 5.19615:")
as = collect(4.90:0.01:5.30)
f0, _ = trace_rays(moving(O, SVector(0.0, 0.0, 0.0)), O, SVector(0.0, 0.0, 0.0), true, d, as; λmax = 20000.0)
summ("static", as, f0)

println("Collinear v = 0.2 (approaching), fine scan, λmax 20000:")
v = SVector(0.0, 0.2, 0.0)
fm, _ = trace_rays(moving(O, v), O, v, true, d, as; λmax = 20000.0)
summ("time-dependent", as, fm)
for vv in (0.05, 0.1, 0.2)
    v = SVector(0.0, vv, 0.0)
    asw = collect(-7.0:0.05:7.0)
    ff, _ = trace_rays(frozen(-v * d, v), -v * d, v, false, d, asw)
    summ(@sprintf("frozen, v = %.2f", vv), asw, ff)
end

println("Transverse v = 0.2 along +x (scan centred on the hole position at light passage, x_h = -60):")
v = SVector(0.2, 0.0, 0.0); sp = -v * d
asw = collect(sp[1] - 7.0:0.05:sp[1] + 7.0)
fm, _ = trace_rays(moving(O, v), O, v, true, d, asw)
summ("time-dependent (a - x_h)", asw .- sp[1], fm)
ff, _ = trace_rays(frozen(sp, v), sp, v, false, d, asw)
summ("frozen at s(-d) (a - x_h)", asw .- sp[1], ff)
for vv in (0.05, 0.1)
    v = SVector(vv, 0.0, 0.0); sp = -v * d
    asw = collect(sp[1] - 7.0:0.05:sp[1] + 7.0)
    fm, _ = trace_rays(moving(O, v), O, v, true, d, asw)
    summ(@sprintf("time-dependent v = %.2f", vv), asw .- sp[1], fm)
    ff, _ = trace_rays(frozen(sp, v), sp, v, false, d, asw)
    summ(@sprintf("frozen v = %.2f", vv), asw .- sp[1], ff)
end
