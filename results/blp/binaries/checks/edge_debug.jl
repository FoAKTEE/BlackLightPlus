include(joinpath(@__DIR__, "fvm_lib.jl"))
using BLP.Spacetime: KerrSchildCartesian, trace_geodesic, ks_radius, null_error
O = SVector(0.0, 0.0, 0.0); v0 = SVector(0.0, 0.0, 0.0); d = 300.0
kerr = KerrSchildCartesian(0.0)
for a in (5.10, 5.15, 5.19, 5.21)
    x0 = SVector(0.0, a, d, 0.0)
    # BLP's own Kerr path
    kk = nullify_covariant(kerr, x0, SVector(0.0, 1.0, 0.0)); yk = vcat(x0, kk / kk[1])
    sk = trace_geodesic(kerr, yk; escape = 450.0, λmax = 20000.0, save_everystep = true)
    rk = [ks_radius(kerr, SVector(u[1], u[2], u[3], u[4])) for u in sk.u]
    # the UserMetric path with the fork formulas at v = 0
    st = UserMetric(moving(O, v0))
    ku = nullify_covariant(st, x0, SVector(0.0, 1.0, 0.0)); yu = vcat(x0, ku / ku[1])
    rhs(y, p, λ) = geodesic_rhs(st, y)
    cap(u, λ, i) = rprime(SVector(u[1], u[2], u[3], u[4]), O, v0, true) - 2.001
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - 450.0
    su = solve(ODEProblem(rhs, yu, (0.0, 20000.0), nothing), Vern7(); reltol = 1e-10, abstol = 1e-12,
               callback = CallbackSet(ContinuousCallback(cap, terminate!), ContinuousCallback(esc, terminate!, nothing)), save_everystep = true)
    ru = [rprime(SVector(u[1], u[2], u[3], u[4]), O, v0, true) for u in su.u]
    @printf("a = %.2f  Kerr path: p0 = %s  r_end %.3f r_min %.3f steps %d |  UserMetric: p0 = %s r_end %.3f r_min %.3f steps %d  null_err %.1e\n",
            a, string(round.(yk[5:8]; digits = 6)), rk[end], minimum(rk), length(rk), string(round.(yu[5:8]; digits = 6)), ru[end], minimum(ru), length(ru), null_error(st, su.u[end]))
end
