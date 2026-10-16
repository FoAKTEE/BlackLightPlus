"""
    BLP

General-relativistic ray tracing, polarized radiative transfer and Monte Carlo
Comptonization in Julia. Conventions: signature (−,+,+,+); G = c = M = 1;
Cartesian Kerr–Schild coordinates; ray state `y = [x^μ; p_μ]` with covariant
momentum; Lorentz-invariant coefficients (j/ν², αν, ρν); Stokes transfer in a
parallel-transported camera frame.

Three tiers, included in dependency order: physics (`Numerics`, `Spacetime`,
`Matter`, `Transfer`, `MonteCarlo`), infrastructure (`Data`, `Execution`) and
pipeline (`Plans`, `Analysis`, `Products`). A product runs on the CPU or on
GPUs according to its `backend` keyword; see `backend!` and `BLP_BACKEND`.
"""
module BLP

include("physics/Numerics/Numerics.jl")
include("physics/Spacetime/Spacetime.jl")
include("physics/Matter/Matter.jl")
include("physics/Transfer/Transfer.jl")
include("physics/MonteCarlo/MonteCarlo.jl")
include("infra/Data/Data.jl")
include("infra/Execution/Execution.jl")
include("pipeline/Plans/Plans.jl")
include("pipeline/Analysis/Analysis.jl")
include("pipeline/Products/Products.jl")

using .Numerics
using .Spacetime
using .Matter
using .Transfer
using .MonteCarlo
using .Data
using .Execution
using .Plans
using .Analysis
using .Products

# Configuration
export CameraConfig, PlasmaConfig, RunConfig, Lanes, MixedLanes, prepare
# Spacetimes
export KerrSchildCartesian, Minkowski, UserMetric, SuperposedKerrSchild, ADMGrid,
       adm_grid_from_metric, z4c_grid_spacetime
# Matter
export AthenaKSnapshot, UniformGrid, SnapshotSeries, FrameWindow, SeriesProvider,
       PolishDoughnut, PageThorneDisk, EquatorialHotspot, EquatorialDisk, UniformSphere
# Transfer modes
export StokesI, IQUV
# Execution
export ThreadedExecutor, PoolExecutor, DevicePool, HostDevice, CUDADevice, select_devices,
       backend, backend!
# Products
export render, render_amr, render_analytic, photon_ring_layers, trace, shadow,
       sensitivity, variants, mc_spectrum
# Results
export Image, Trajectories, Spectrum, ScatterResult, GRMHDSpectrumResult
# Analysis and I/O
export write_npy, visamp, visamp_minima, radon_cut, tukey_window, lensing_band, critical_curve

end # module
