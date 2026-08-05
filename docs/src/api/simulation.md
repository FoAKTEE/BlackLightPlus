# Simulation API

The `Simulation` module reads external simulation data: AthenaK GRMHD
snapshots [Stone2026](@citep) in `.athdf` and `.bin` form, Z4c
numerical-relativity dumps [Zhu2025](@citep), and time series of snapshots for
slow-light rendering.

## AthenaK GRMHD input

```@docs
athenak_bin_snapshot
read_athenak_bin
```

## AMR meshblock indexing

A dense finest-level leaf-index grid that turns block location into an $O(1)$
lookup. The same structure is uploaded to the device for native GPU AMR
sampling.

```@docs
LeafTable
leaf_table
meshblock_index
```

## Z4c numerical relativity

Maps the conformal Z4c variables to physical ADM fields
($\gamma_{ij} = \tilde g_{ij}/\chi$, with $\alpha$ and $\beta^i$ passed
through) and builds an [`ADMGridSpacetime`](@ref).

```@docs
resample_z4c_adm
z4c_grid_spacetime
```

## Snapshot series and slow light

```@docs
SnapshotSeries
synthetic_series
bracketing_pair
get_frame
sample_primitives_t
scale_density
```
