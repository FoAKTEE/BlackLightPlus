# The Monte Carlo packet: `w` physical photons sharing one null geodesic (dN = w dN_s).
# Signature (−,+,+,+), G = c = M = 1; the momentum is covariant, as in the ray state
# y = [x^μ; p_μ], so p_t is conserved on a stationary spacetime and E = −p_t.

"""
    Superphoton

One Monte Carlo superphoton, `isbits`, so a `Vector{Superphoton}` is a device buffer.

Fields: the position `x::SVector{4,Float64}` (coordinates, time first), the covariant
wave vector `p::SVector{4,Float64}` in mₑc² units, the weight `w` (physical photons per
packet), the conserved energy `E = −p_t` (mₑc²), the absorption and scattering optical
depths accumulated so far `tau_abs` and `tau_scatt`, the scattering count
`nscatt::Int32`, and diagnostics of the emission event: the emission coordinates
`x1i`, `x2i` (x¹, x² of the emission point) and the plasma there, `ne0` (cm⁻³),
`thetae0` and `b0` (G).
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
    Superphoton(x, p, w; tau_abs = 0, tau_scatt = 0, nscatt = 0,
                x1i = x[2], x2i = x[3], ne0 = 0, thetae0 = 0, b0 = 0)

The superphoton at `x` with covariant momentum `p` and weight `w`; `E = −p_t` is
derived from `p`. Every number is converted to the `Float64`/`Int32` field type.
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

"Coordinate position x^μ of the superphoton."
@inline photon_position(sp::Superphoton) = sp.x
"Covariant wave vector p_μ (mₑc² units)."
@inline photon_momentum(sp::Superphoton) = sp.p
"Statistical weight: physical photons per packet."
@inline photon_weight(sp::Superphoton) = sp.w
"Conserved energy E = −p_t (mₑc² units)."
@inline photon_energy(sp::Superphoton) = sp.E
"Number of scatterings undergone."
@inline photon_nscatt(sp::Superphoton) = sp.nscatt
