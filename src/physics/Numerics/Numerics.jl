"""
    Numerics

Number-type-generic building blocks shared by every tier: CGS constants, units, ray-state accessors, interpolation stencils, exact integer-limb sums, the Philox counter RNG and the special functions of the synchrotron fits. Pure, allocation-free, `isbits` values only.
"""
module Numerics

using StaticArrays
import ForwardDiff
import Atomix

include("constants.jl")
include("units.jl")
include("raystate.jl")
include("stencils.jl")
include("exactsum.jl")
include("philox.jl")
include("special.jl")

export CL, EE, ME, MP, KB, HPL, GNEWT, MSUN, SQRT2, SIGMA_THOMSON
export Units
export RayState, position, momentum, raystate
export primal, corner_weights, lerp8, cr_weights, cr_axis, cr_sample, loglinear_at
export XSUM_LIMB_BITS, XSUM_NLIMBS, XSUM_NSLOT, XSUM_MAX_ADDENDS,
       xsum_parts, xsum_add!, xsum_atomic_add!, xsum_round, xsum_zeros, xsum_round_all
export philox4x32, philox_rand, MCRng, mc_uniform, mc_gauss, sample_direction,
       standard_gamma, chi_square, MCStreamLayout, stream_id, stream_layer,
       primary_rng, child_rng
export bessel_i0, bessel_i1, bessel_k0, bessel_k1, bessel_k2, harmonic_bridge,
       hypergeometric_2f1

end # module
