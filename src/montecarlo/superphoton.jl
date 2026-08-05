# Monte Carlo — superphoton.jl
#
# Superphoton state carried through generation, tracking and scattering: a
# Monte Carlo packet of `w` physical photons sharing one geodesic (dN = w dN_s).
# Conventions: signature (-,+,+,+), G = c = M = 1; the momentum is stored
# COVARIANT (p_mu), matching the geodesic state y = [x^mu; p_mu] so p_t is
# conserved exactly for stationary metrics, and the photon energy is E = -p_t.
# The struct is deliberately `isbits` so an array of `Superphoton` is a
# contiguous, GPU-uploadable buffer usable inside a KernelAbstractions kernel.

"""
    Superphoton

Monte Carlo superphoton state (`isbits`).

Fields
- `x::SVector{4,Float64}`      : coordinate position x^mu.
- `p::SVector{4,Float64}`      : COVARIANT wavevector p_mu (m_e c^2 units).
- `w::Float64`                 : statistical weight (photons per packet, >> 1).
- `E::Float64`                 : conserved energy `-p_t`.
- `tau_abs::Float64`           : accumulated absorption optical depth.
- `tau_scatt::Float64`         : accumulated scattering optical depth.
- `nscatt::Int32`              : number of scatterings undergone.
- `x1i,x2i::Float64`           : emission-point coordinates (diagnostics).
- `ne0,thetae0,b0::Float64`    : plasma state at emission (n_e, Theta_e, B) (diagnostics).
"""
struct Superphoton
    x::SVector{4,Float64}
    p::SVector{4,Float64}
    w::Float64
    E::Float64
    tau_abs::Float64
    tau_scatt::Float64
    nscatt::Int32
    x1i::Float64
    x2i::Float64
    ne0::Float64
    thetae0::Float64
    b0::Float64
end

"""
    Superphoton(x, p, w; tau_abs=0, tau_scatt=0, nscatt=0,
                x1i=x[2], x2i=x[3], ne0=0, thetae0=0, b0=0)

Convenience constructor. `E = -p_t` is derived from the covariant momentum, and
the emission-point diagnostics default to the current position (`x1i`, `x2i`) and
zero plasma state. All numeric arguments are widened to `Float64` / `Int32` so the
result is `isbits`.
"""
@inline function Superphoton(x::SVector{4}, p::SVector{4}, w::Real;
                             tau_abs::Real = 0.0, tau_scatt::Real = 0.0,
                             nscatt::Integer = 0,
                             x1i::Real = x[2], x2i::Real = x[3],
                             ne0::Real = 0.0, thetae0::Real = 0.0, b0::Real = 0.0)
    return Superphoton(SVector{4,Float64}(x), SVector{4,Float64}(p),
                       Float64(w), -Float64(p[1]),
                       Float64(tau_abs), Float64(tau_scatt), Int32(nscatt),
                       Float64(x1i), Float64(x2i),
                       Float64(ne0), Float64(thetae0), Float64(b0))
end

# ---------------------------------------------------------------- accessors
# MC-specific names so nothing collides with the exported `energy` (geodesics).

"Coordinate position x^mu."
@inline photon_position(sp::Superphoton) = sp.x
"Covariant wavevector p_mu."
@inline photon_momentum(sp::Superphoton) = sp.p
"Statistical weight (photons per packet)."
@inline photon_weight(sp::Superphoton) = sp.w
"Conserved photon energy E = -p_t."
@inline photon_energy(sp::Superphoton) = sp.E
"Number of scatterings undergone."
@inline photon_nscatt(sp::Superphoton) = sp.nscatt
