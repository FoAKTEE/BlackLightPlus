"""
    Numerics

Number-type-generic building blocks shared by every tier: CGS constants, units, ray-state accessors, interpolation stencils, exact integer-limb sums, the Philox counter RNG and the special functions of the synchrotron fits. Pure, allocation-free, `isbits` values only.
"""
module Numerics

using StaticArrays

include("constants.jl")

export CL, EE, ME, MP, KB, HPL, GNEWT, MSUN, SQRT2, SIGMA_THOMSON

end # module
