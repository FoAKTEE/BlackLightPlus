"""
    Simulation

AthenaK snapshot reading, meshblock sampling, and the normal-frame
primitive → four-vector transform (AthenaK: Stone et al. 2024).
"""
module Simulation

using StaticArrays
using LinearAlgebra
using HDF5
using ..Constants: CL, GNEWT, MP, ME
using ..Geodesics: ADMGridSpacetime

export primitive_four_vectors, pgas_from_eint,
       AthenaKSnapshot, sample_primitives, units,
       read_athenak_bin, athenak_bin_snapshot, resample_z4c_adm, z4c_grid_spacetime,
       sample_var,
       SnapshotSeries, synthetic_series, scale_density,
       bracketing_pair, get_frame, sample_primitives_t,
       LeafTable, leaf_table, meshblock_index

include("primitives.jl")        # primitive -> four-vector transform
include("athenak_reader.jl")    # AthenaK .athdf reader + sampler
include("athenak_z4c.jl")       # AthenaK .bin Z4c AMR reader + ADM grid builder
include("time_series.jl")       # slow light: dump series + time-blended sampler

end # module
