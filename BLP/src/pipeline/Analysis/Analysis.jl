"""
    Analysis

Image-plane analysis: the block refinement indicators of adaptive rendering, the
lensing-band geometry of photon-ring imaging on the Bardeen screen (apparent horizon,
band edges, band grids, polygon area and membership), and interferometric visibility
amplitudes (Radon cut, Tukey window, |V| against baseline in Gλ). Host code; the screen
unit is M and image angles are in μas.
"""
module Analysis

using FFTW: fft, fftshift
import JacobiElliptic
using ..Spacetime: critical_curve, conserved_quantities, radial_roots, angular_turning_points,
                   kerr_horizon, kerr_radial_potential

include("refinement.jl")
include("lensing_bands.jl")
include("interferometry.jl")

# block refinement
export RefinementIndicator, RelativeGradient, RelativeLaplacian, AbsoluteGradient, AnyOf,
       refinement_indicator, needs_refinement
# lensing bands
export polygon_area, in_polygon, apparent_horizon, lensing_band, band_grid
# interferometry
export MUAS_TO_RAD, tukey_window, radon_cut, visamp, visamp_minima

end # module
