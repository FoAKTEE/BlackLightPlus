"""
    Data

Containers and files: AthenaK snapshots as feature-fastest fluid blocks with eager locate
tables, the `.athdf` and `.bin` readers, uniform grids, snapshot series with their
two-frame windows, and the npy writer. Every grid-backed `PlasmaProvider` lives here; the
sampling structs are `isbits` after `Adapt.adapt`.
"""
module Data

using StaticArrays
using HDF5: h5open, read_attribute, attributes
import Adapt
using ..Numerics: corner_weights, lerp8
using ..Spacetime: ADMGrid
using ..Matter: PlasmaPrimitives, GridProvider
import ..Matter: primitives, domain_radius

include("fluid_blocks.jl")
include("athenak_hdf5.jl")
include("athenak_bin.jl")
include("uniform_grid.jl")
include("series.jl")
include("npy.jl")

# fluid blocks and locate strategies
export PRIM_NAMES, FluidBlocks, LeafTable, LevelHash, BlockScan, leaf_table, level_hash,
       build_locate, with_locate, locate, block_sample, same_mesh
# AthenaK files
export AthenaKSnapshot, share_mesh, AthenaKBin, read_athenak_bin, sample_var,
       resample_z4c_adm, z4c_grid_spacetime, athenak_bin_snapshot
# uniform grids
export UniformGrid, grid_sample, resample, scale_density
# series and windows
export SnapshotSeries, synthetic_series, bracketing_pair, get_frame, blend, FrameWindow,
       window, advance, SeriesProvider
# output
export write_npy

end # module
