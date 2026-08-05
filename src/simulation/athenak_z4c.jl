# Simulation — athenak_z4c.jl
#
# AthenaK `.bin` AMR reader for Z4c numerical-spacetime dumps (AthenaK: Stone
# et al. 2024), the Z4c -> ADM variable conversion, and a resample of the 10
# ADM fields onto a uniform Cartesian box feeding `ADMGridSpacetime`.
#
# z4c -> ADM (AthenaK z4c/z4c_adm.cpp Z4cToADM, chi_psi_power = -4):
#   gamma_ij = g̃_ij / chi   (psi4 = chi^(4/chi_psi_power) = 1/chi),
#   alpha = z4c_alpha,  beta^i = z4c_beta{x,y,z}   (chi floored at 1e-12).

using StaticArrays

"An AthenaK `.bin` AMR snapshot: per-meshblock bounds + variables (var-major)."
struct AthenaKBin
    n_mb::Int
    nx1::Int; nx2::Int; nx3::Int
    var_names::Vector{String}
    level::Vector{Int}
    x1min::Vector{Float64}; x1max::Vector{Float64}
    x2min::Vector{Float64}; x2max::Vector{Float64}
    x3min::Vector{Float64}; x3max::Vector{Float64}
    data::Array{Float32,5}      # (nx1, nx2, nx3, n_vars, n_mb)  — i fastest
    time::Float64
end

"""
    read_athenak_bin(filename) -> AthenaKBin

Read an "Athena binary output version=1.1" `.bin` dump: ASCII header ending in
"header offset=<H>", an <H>-byte parameter dump, then per meshblock
Int32[10] = (ois,oie,ojs,oje,oks,oke, lx1,lx2,lx3, level), Real[6] block bounds
(Real = the header's "size of location": 8, or 4 for a single-precision build),
and Float32[n_vars·nx3·nx2·nx1] cell data (i fastest, then j, k, var).
"""
function read_athenak_bin(filename::AbstractString)
    io = open(filename, "r")
    try
        first = readline(io)
        startswith(first, "Athena binary output version=1.1") ||
            error("not an Athena binary v1.1 file: $first")
        n_vars = 0; var_names = String[]; header_offset = -1; time = 0.0
        loc_size = 8; var_size = 4                    # writer defaults: sizeof(Real), sizeof(float)
        while header_offset < 0
            line = readline(io)
            if startswith(line, "  time=")
                time = parse(Float64, strip(line[8:end]))
            elseif startswith(line, "  size of location=")
                loc_size = parse(Int, strip(line[20:end]))
            elseif startswith(line, "  size of variable=")
                var_size = parse(Int, strip(line[20:end]))
            elseif startswith(line, "  number of variables=")
                n_vars = parse(Int, strip(line[23:end]))
            elseif startswith(line, "  variables:")
                var_names = String.(split(strip(line[13:end])))
            elseif startswith(line, "  header offset=")
                header_offset = parse(Int, strip(line[17:end]))
            end
        end
        length(var_names) == n_vars || error("variable-count mismatch")
        loc_size == 8 || loc_size == 4 ||
            error("unsupported 'size of location=$loc_size' (expected 4 or 8)")
        var_size == 4 || error("unsupported 'size of variable=$var_size' (expected 4)")
        skip(io, header_offset)                       # skip the parameter dump
        data_start = position(io)

        # first block header -> uniform block dims
        hdr = Vector{Int32}(undef, 10); read!(io, hdr)
        nx1 = Int(hdr[2] - hdr[1] + 1)
        nx2 = Int(hdr[4] - hdr[3] + 1)
        nx3 = Int(hdr[6] - hdr[5] + 1)
        cells = nx1 * nx2 * nx3
        per_mb = 10 * 4 + 6 * loc_size + cells * n_vars * var_size
        seekend(io); file_size = position(io)
        (file_size - data_start) % per_mb == 0 ||
            error("meshblock size mismatch: $(file_size - data_start) data bytes " *
                  "not a multiple of $per_mb per block (truncated or wrong-layout file)")
        n_mb = (file_size - data_start) ÷ per_mb

        level = Vector{Int}(undef, n_mb)
        x1min = Vector{Float64}(undef, n_mb); x1max = similar(x1min)
        x2min = similar(x1min); x2max = similar(x1min)
        x3min = similar(x1min); x3max = similar(x1min)
        # store (nx1, nx2, nx3, n_vars, n_mb): on disk i fastest then j,k,var
        alldata = Array{Float32,5}(undef, nx1, nx2, nx3, n_vars, n_mb)
        # block bounds are sizeof(Real) on disk (4 or 8 bytes, per the header)
        bnds = loc_size == 8 ? Vector{Float64}(undef, 6) : Vector{Float32}(undef, 6)
        seek(io, data_start)
        for b in 1:n_mb
            read!(io, hdr)
            level[b] = Int(hdr[10])
            read!(io, bnds)
            x1min[b], x1max[b] = bnds[1], bnds[2]
            x2min[b], x2max[b] = bnds[3], bnds[4]
            x3min[b], x3max[b] = bnds[5], bnds[6]
            blk = @view alldata[:, :, :, :, b]
            read!(io, blk)                            # fills i,j,k,var in disk order
        end
        return AthenaKBin(n_mb, nx1, nx2, nx3, var_names, level,
                          x1min, x1max, x2min, x2max, x3min, x3max, alldata, time)
    finally
        close(io)
    end
end

_varidx(snap::AthenaKBin, name) = findfirst(==(name), snap.var_names)

"Finest meshblock containing (x,y,z); returns 0 if none (last/finest wins)."
function _locate_block(s::AthenaKBin, x, y, z)
    hit = 0; best = -1
    @inbounds for b in 1:s.n_mb
        (s.x1min[b] <= x <= s.x1max[b] && s.x2min[b] <= y <= s.x2max[b] &&
         s.x3min[b] <= z <= s.x3max[b]) || continue
        if s.level[b] > best
            best = s.level[b]; hit = b
        end
    end
    return hit
end

# Precomputed trilinear stencil at one point: owning block `b`, lower corner
# indices, and fractional weights. Shared across the 11 fields read at the same
# (x,y,z) to avoid re-locating per variable. `b == 0` marks an off-mesh point.
struct _Stencil
    b::Int
    i0::Int; j0::Int; k0::Int
    ti::Float64; tj::Float64; tk::Float64
end

"Locate the owning block at (x,y,z) and build its trilinear stencil once."
@inline function _point_stencil(s::AthenaKBin, x, y, z)
    b = _locate_block(s, x, y, z)
    b == 0 && return _Stencil(0, 1, 1, 1, 0.0, 0.0, 0.0)
    dx1 = (s.x1max[b] - s.x1min[b]) / s.nx1
    dx2 = (s.x2max[b] - s.x2min[b]) / s.nx2
    dx3 = (s.x3max[b] - s.x3min[b]) / s.nx3
    fi = (x - s.x1min[b]) / dx1 - 0.5 + 1        # continuous 1-based index
    fj = (y - s.x2min[b]) / dx2 - 0.5 + 1
    fk = (z - s.x3min[b]) / dx3 - 0.5 + 1
    i0 = clamp(floor(Int, fi), 1, s.nx1 - 1); ti = clamp(fi - i0, 0.0, 1.0)
    j0 = clamp(floor(Int, fj), 1, s.nx2 - 1); tj = clamp(fj - j0, 0.0, 1.0)
    k0 = clamp(floor(Int, fk), 1, s.nx3 - 1); tk = clamp(fk - k0, 0.0, 1.0)
    return _Stencil(b, i0, j0, k0, ti, tj, tk)
end

"Trilinear-gather variable `v` (1-based col) from a precomputed stencil."
@inline function _sample_stencil(s::AthenaKBin, st::_Stencil, v::Int)
    b = st.b; i0 = st.i0; j0 = st.j0; k0 = st.k0
    ti = st.ti; tj = st.tj; tk = st.tk
    d = s.data
    @inline g(i, j, k) = Float64(d[i, j, k, v, b])
    c00 = g(i0, j0, k0) * (1 - ti) + g(i0 + 1, j0, k0) * ti
    c10 = g(i0, j0 + 1, k0) * (1 - ti) + g(i0 + 1, j0 + 1, k0) * ti
    c01 = g(i0, j0, k0 + 1) * (1 - ti) + g(i0 + 1, j0, k0 + 1) * ti
    c11 = g(i0, j0 + 1, k0 + 1) * (1 - ti) + g(i0 + 1, j0 + 1, k0 + 1) * ti
    c0 = c00 * (1 - tj) + c10 * tj
    c1 = c01 * (1 - tj) + c11 * tj
    return c0 * (1 - tk) + c1 * tk
end

"Trilinear-sample variable `v` (1-based col) at (x,y,z); NaN if off-mesh."
function sample_var(s::AthenaKBin, v::Int, x, y, z)
    st = _point_stencil(s, x, y, z)
    st.b == 0 && return NaN
    return _sample_stencil(s, st, v)
end

const _Z4C_CHI_PSI_POWER = -4.0
const _CHI_FLOOR = 1.0e-12

"Resolve the z4c variable columns needed for the ADM reconstruction."
function _z4c_cols(s::AthenaKBin)
    idx(n) = (i = _varidx(s, n); i === nothing && error("missing z4c var $n"); i)
    return (chi=idx("z4c_chi"), gxx=idx("z4c_gxx"), gxy=idx("z4c_gxy"), gxz=idx("z4c_gxz"),
            gyy=idx("z4c_gyy"), gyz=idx("z4c_gyz"), gzz=idx("z4c_gzz"),
            alpha=idx("z4c_alpha"), betax=idx("z4c_betax"), betay=idx("z4c_betay"),
            betaz=idx("z4c_betaz"))
end

"""
    resample_z4c_adm(snap, half, n) -> (axis, alpha, beta, gamma)

Resample the physical ADM fields from a Z4c `.bin` snapshot onto a uniform
`n×n×n` box `[-half, half]³`: returns the axis range, `alpha[n,n,n]`,
`beta[3,n,n,n]` (β^i), and `gamma[6,n,n,n]` (gxx,gxy,gxz,gyy,gyz,gzz), with
`gamma_ij = g̃_ij / chi`. Off-mesh points get Minkowski (alpha=1, beta=0,
gamma=δ). Threaded over the grid.
"""
function resample_z4c_adm(s::AthenaKBin, half::Float64, n::Int)
    c = _z4c_cols(s)
    axis = range(-half, half; length=n)
    alpha = Array{Float64,3}(undef, n, n, n)
    beta = Array{Float64,4}(undef, 3, n, n, n)
    gamma = Array{Float64,4}(undef, 6, n, n, n)
    Threads.@threads for k in 1:n
        z = axis[k]
        for j in 1:n, i in 1:n
            x = axis[i]; y = axis[j]
            # Locate the owning block + build the trilinear stencil once, then
            # gather all 11 ADM fields from it.
            st = _point_stencil(s, x, y, z)
            chi = st.b == 0 ? NaN : _sample_stencil(s, st, c.chi)
            if isnan(chi)
                alpha[i, j, k] = 1.0
                beta[1, i, j, k] = beta[2, i, j, k] = beta[3, i, j, k] = 0.0
                gamma[1, i, j, k] = gamma[4, i, j, k] = gamma[6, i, j, k] = 1.0
                gamma[2, i, j, k] = gamma[3, i, j, k] = gamma[5, i, j, k] = 0.0
                continue
            end
            chi = max(chi, _CHI_FLOOR)
            psi4 = chi^(4.0 / _Z4C_CHI_PSI_POWER)          # = 1/chi
            alpha[i, j, k] = _sample_stencil(s, st, c.alpha)
            beta[1, i, j, k] = _sample_stencil(s, st, c.betax)
            beta[2, i, j, k] = _sample_stencil(s, st, c.betay)
            beta[3, i, j, k] = _sample_stencil(s, st, c.betaz)
            gamma[1, i, j, k] = psi4 * _sample_stencil(s, st, c.gxx)
            gamma[2, i, j, k] = psi4 * _sample_stencil(s, st, c.gxy)
            gamma[3, i, j, k] = psi4 * _sample_stencil(s, st, c.gxz)
            gamma[4, i, j, k] = psi4 * _sample_stencil(s, st, c.gyy)
            gamma[5, i, j, k] = psi4 * _sample_stencil(s, st, c.gyz)
            gamma[6, i, j, k] = psi4 * _sample_stencil(s, st, c.gzz)
        end
    end
    return axis, alpha, beta, gamma
end

"""
    z4c_grid_spacetime(binfile, half, n; alpha_min=0.15) -> ADMGridSpacetime

Read an AthenaK Z4c `.bin` dump, resample the ADM fields onto a uniform `n³` box
`[-half, half]³`, and return an `ADMGridSpacetime` for geodesic tracing.
`alpha_min` is the lapse-collapse inner boundary (0.15 sits outside the horizons).
Frozen-slice approximation: one time slice is traced with `∂ₜg_μν = 0` (the ADM
time-derivative terms are dropped), valid only while the light-crossing time
`≈ 2·half` is ≪ the spacetime's evolution timescale (BBH orbital period).
"""
function z4c_grid_spacetime(binfile::AbstractString, half::Float64, n::Int;
                            alpha_min::Real=0.15)
    @warn "z4c_grid_spacetime: FROZEN-SLICE metric (∂ₜg = 0) from one .bin slice; " *
          "valid only while light-crossing time (~2·half = $(2half)) ≪ the spacetime " *
          "evolution timescale (BBH orbital period). No ∂ₜ(α,β,γ) terms." maxlog=1
    s = read_athenak_bin(binfile)
    axis, alpha, beta, gamma = resample_z4c_adm(s, half, n)
    data = Array{Float64,4}(undef, 10, n, n, n)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        data[1, i, j, k] = alpha[i, j, k]
        data[2, i, j, k] = beta[1, i, j, k]
        data[3, i, j, k] = beta[2, i, j, k]
        data[4, i, j, k] = beta[3, i, j, k]
        data[5, i, j, k] = gamma[1, i, j, k]   # gxx
        data[6, i, j, k] = gamma[2, i, j, k]   # gxy
        data[7, i, j, k] = gamma[3, i, j, k]   # gxz
        data[8, i, j, k] = gamma[4, i, j, k]   # gyy
        data[9, i, j, k] = gamma[5, i, j, k]   # gyz
        data[10, i, j, k] = gamma[6, i, j, k]  # gzz
    end
    return ADMGridSpacetime(axis, axis, axis, data; alpha_min=alpha_min), s
end

# ---------------------------------------------------------------- GRMHD-fluid .bin adapter

"""
    athenak_bin_snapshot(binfile; gamma) -> AthenaKSnapshot

Adapt an AthenaK GRMHD fluid `.bin` dump (`mhd_w_bcc` output group) into an
[`AthenaKSnapshot`](@ref). `dens, velx, vely, velz, bcc1..3` pass through
bit-identically; `press → eint = press/(gamma-1)` in Float64, rounded once to
Float32; a `temperature` column is dropped. Non-finite cells (isolated c2p
failures) are zeroed so validity guards skip them. Logical locations are derived
from block bounds (`lx = (x_min - domain_min)/width`, exact for AthenaK's
power-of-two layout, checked integral per block).
"""
function athenak_bin_snapshot(binfile::AbstractString; gamma::Real)
    s = read_athenak_bin(binfile)
    col(n) = (i = _varidx(s, n); i === nothing && error("missing GRMHD var $n in $binfile " *
                                                        "(vars: $(s.var_names))"); i)
    cols = map(col, ("dens", "velx", "vely", "velz", "press", "bcc1", "bcc2", "bcc3"))
    ni, nj, nk, nmb = s.nx1, s.nx2, s.nx3, s.n_mb
    gm1 = Float64(gamma) - 1.0
    src = s.data                                         # [i,j,k,var,mb]
    data = Array{Float32,5}(undef, ni, nj, nk, nmb, 8)   # [i,j,k,mb,prim], _PRIM_NAMES order
    nbad = Threads.Atomic{Int}(0)
    Threads.@threads for m in 1:nmb
        bad = 0
        @inbounds for (prim, v) in enumerate(cols), k in 1:nk, j in 1:nj, i in 1:ni
            val = src[i, j, k, v, m]
            isfinite(val) || (val = 0.0f0; bad += 1)
            data[i, j, k, m, prim] = prim == 5 ? Float32(Float64(val) / gm1) : val
        end
        bad > 0 && Threads.atomic_add!(nbad, bad)
    end
    nbad[] > 0 && @warn "athenak_bin_snapshot: zeroed $(nbad[]) non-finite cell values " *
                        "(isolated c2p failures; zeroed cells emit nothing)" maxlog=1
    ext_lo = [SVector(s.x1min[m], s.x2min[m], s.x3min[m]) for m in 1:nmb]
    ext_hi = [SVector(s.x1max[m], s.x2max[m], s.x3max[m]) for m in 1:nmb]
    dx     = [(ext_hi[m] - ext_lo[m]) ./ SVector(ni, nj, nk) for m in 1:nmb]
    x_left = [ext_lo[m] + dx[m] / 2 for m in 1:nmb]
    dom = SVector(minimum(s.x1min), minimum(s.x2min), minimum(s.x3min))
    ll = Matrix{Int}(undef, 3, nmb)
    for m in 1:nmb, ax in 1:3
        f = (ext_lo[m][ax] - dom[ax]) / (ext_hi[m][ax] - ext_lo[m][ax])
        ll[ax, m] = round(Int, f)
        abs(f - ll[ax, m]) <= 1e-6 ||
            error("athenak_bin_snapshot: block $m axis $ax off the power-of-two " *
                  "logical grid (fractional index $f) — corrupt or unsupported mesh")
    end
    return AthenaKSnapshot(data, ext_lo, ext_hi, x_left, dx, s.level, ll,
                           collect(String, _PRIM_NAMES), ni, nj, nk, nmb)
end
