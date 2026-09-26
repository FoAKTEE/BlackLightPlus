# r10_trapezoid.jl — transmission of one absorbing, non-emitting segment through RAPTOR's
# native polarized step (check_stiffness -> trapezoid) and the LD85 step (M11), against
# exp(-tau): S = (1, 0, 0, 0), alpha_I only, tau = 0.1 ... 1000.
# Usage: julia --project=. r10_trapezoid.jl <libraptor_pol_step.so> [out.json]
# Exit 0 iff the LD85 step equals exp(-tau) to 1e-12 (relative, where exp(-tau) > 1e-300).
using Printf
include(joinpath(@__DIR__, "jsonw.jl"))
h = Libc.Libdl.dlopen(ARGS[1]); f = Libc.Libdl.dlsym(h, :raptor_pol_step)
taus = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
rows = Dict{String,Any}[]
ok = true
for tau in taus
    out = Dict{String,Any}("tau" => tau, "exp_minus_tau" => exp(-tau))
    for (solver, name) in ((0, "native"), (1, "ld85"))
        coef = zeros(11); coef[8] = tau        # alpha_I, dl * C = 1
        S = [1.0, 0.0, 0.0, 0.0]
        st = ccall(f, Cint, (Cint, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64}), solver, coef, 1.0, 1.0, S)
        out[name] = S[1]; solver == 0 && (out["native_stiff"] = st)
    end
    global ok &= abs(out["ld85"] / exp(-tau) - 1) <= 1e-12 || exp(-tau) < 1e-300
    push!(rows, out)
    @printf("tau %7.1f  exp(-tau) %.3e  native %+.6e (%s)  (1 - tau/2)/(1 + tau/2) %+.6e  LD85 %.3e\n", tau, exp(-tau),
            out["native"], out["native_stiff"] == 1 ? "trapezoid" : "RK4", (1 - tau / 2) / (1 + tau / 2), out["ld85"])
end
length(ARGS) > 1 && write_json(ARGS[2], Dict("segments" => rows, "pass" => ok))
exit(ok ? 0 : 1)
