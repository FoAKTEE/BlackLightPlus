# Code-unit to cgs conversions of a GRMHD model (geometrized G = c = M = 1).

"""
    Units{T}

GRMHD code-unit to cgs conversions for a black hole of mass `m_bh` and a
density mass scale `mass_scale` (both in g):

- `L_unit = G m_bh / c²` (cm), `T_unit = L_unit / c` (s);
- `dens_unit = mass_scale / L_unit³` (g cm⁻³);
- `Ne_unit = dens_unit / (m_p + m_e)` (cm⁻³);
- `B_unit = c √(4π dens_unit)` (G);
- `e_unit = dens_unit c²` (erg cm⁻³).

`T` is the promoted type of the six fields: a `ForwardDiff.Dual` mass scale
makes every field a `Dual`, the mass-scale-independent `L_unit` and `T_unit`
with zero partials.
"""
struct Units{T}
    L_unit::T
    T_unit::T
    dens_unit::T
    Ne_unit::T
    B_unit::T
    e_unit::T
end

"""
    Units(m_bh_cgs, mass_scale_cgs) -> Units

The unit set of a hole of mass `m_bh_cgs` (g) and density mass scale
`mass_scale_cgs` (g). Generic in the number type of either argument.
"""
@inline function Units(m_bh_cgs, mass_scale_cgs)
    L_unit    = GNEWT * m_bh_cgs / CL^2
    T_unit    = L_unit / CL
    dens_unit = mass_scale_cgs / L_unit^3
    Ne_unit   = dens_unit / (MP + ME)
    B_unit    = CL * sqrt(4π * dens_unit)
    e_unit    = dens_unit * CL^2
    T = promote_type(typeof(L_unit), typeof(T_unit), typeof(dens_unit), typeof(Ne_unit),
                     typeof(B_unit), typeof(e_unit))
    return Units{T}(L_unit, T_unit, dens_unit, Ne_unit, B_unit, e_unit)
end
