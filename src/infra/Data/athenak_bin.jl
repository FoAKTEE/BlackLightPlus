# AthenaK `.bin` dumps: the reader, a trilinear sampler of single variables, the Z4c to
# ADM conversion resampled onto an `ADMGrid`, and the GRMHD-fluid adapter to an
# `AthenaKSnapshot`. Z4c to ADM (AthenaK `z4c/z4c_adm.cpp` `Z4cToADM`,
# `chi_psi_power = -4`): `γ_ij = g̃_ij / χ` (ψ⁴ = χ^(4/chi_psi_power) = 1/χ, χ floored at
# 1e-12), `α = z4c_alpha`, `β^i = z4c_beta{x,y,z}`. AthenaK: Stone et al. (2024).

"""
    AthenaKBin

An AthenaK `.bin` AMR dump. Fields: `n_mb` blocks of `nx1 × nx2 × nx3` cells;
`var_names`; per-block AMR `level` and face bounds `x1min, x1max, x2min, x2max, x3min,
x3max`; `data`, the `(nx1, nx2, nx3, n_vars, n_mb)` `Float32` cell values (`i` fastest,
as on disk); `time`, the dump time (code units).
"""
struct AthenaKBin
    n_mb::Int
    nx1::Int; nx2::Int; nx3::Int
    var_names::Vector{String}
    level::Vector{Int}
    x1min::Vector{Float64}; x1max::Vector{Float64}
    x2min::Vector{Float64}; x2max::Vector{Float64}
    x3min::Vector{Float64}; x3max::Vector{Float64}
    data::Array{Float32,5}
    time::Float64
end

"""
    read_athenak_bin(filename) -> AthenaKBin

Read an "Athena binary output version=1.1" dump: an ASCII header ending in
`header offset=<H>`, an `H`-byte parameter dump, then per block `Int32[10] = (ois, oie,
ojs, oje, oks, oke, lx1, lx2, lx3, level)`, `Real[6]` face bounds (`Real` has the
header's `size of location`, 8 or 4 bytes) and `Float32[n_vars nx3 nx2 nx1]` cell data
(`i` fastest, then `j`, `k`, variable). Errors on another version, a variable size other
than 4 bytes, or a data section that is not a whole number of blocks.
"""
function read_athenak_bin(filename::AbstractString)
    return open(filename, "r") do io
        first = readline(io)
        startswith(first, "Athena binary output version=1.1") ||
            error("not an Athena binary v1.1 file: $first")
        h = _bin_header(io)
        skip(io, h.header_offset)
        data_start = position(io)
        hdr = Vector{Int32}(undef, 10)
        read!(io, hdr)
        nx1 = Int(hdr[2] - hdr[1] + 1)
        nx2 = Int(hdr[4] - hdr[3] + 1)
        nx3 = Int(hdr[6] - hdr[5] + 1)
        per_mb = 10 * 4 + 6 * h.loc_size + nx1 * nx2 * nx3 * h.n_vars * h.var_size
        seekend(io)
        nbytes = position(io) - data_start
        nbytes % per_mb == 0 ||
            error("meshblock size mismatch: $nbytes data bytes not a multiple of $per_mb " *
                  "per block (truncated or wrong-layout file)")
        seek(io, data_start)
        _read_bin_blocks(io, h, nx1, nx2, nx3, nbytes ÷ per_mb)
    end
end

# The header lines after the version line, up to and including `header offset`.
function _bin_header(io)
    n_vars = 0; var_names = String[]; header_offset = -1; time = 0.0
    loc_size = 8; var_size = 4
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
    return (n_vars = n_vars, var_names = var_names, header_offset = header_offset,
            time = time, loc_size = loc_size, var_size = var_size)
end

function _read_bin_blocks(io, h, nx1, nx2, nx3, n_mb)
    level = Vector{Int}(undef, n_mb)
    x1min = Vector{Float64}(undef, n_mb); x1max = similar(x1min)
    x2min = similar(x1min); x2max = similar(x1min)
    x3min = similar(x1min); x3max = similar(x1min)
    alldata = Array{Float32,5}(undef, nx1, nx2, nx3, h.n_vars, n_mb)
    hdr = Vector{Int32}(undef, 10)
    bnds = h.loc_size == 8 ? Vector{Float64}(undef, 6) : Vector{Float32}(undef, 6)
    for b in 1:n_mb
        read!(io, hdr)
        level[b] = Int(hdr[10])
        read!(io, bnds)
        x1min[b], x1max[b] = bnds[1], bnds[2]
        x2min[b], x2max[b] = bnds[3], bnds[4]
        x3min[b], x3max[b] = bnds[5], bnds[6]
        read!(io, @view alldata[:, :, :, :, b])
    end
    return AthenaKBin(n_mb, nx1, nx2, nx3, h.var_names, level,
                      x1min, x1max, x2min, x2max, x3min, x3max, alldata, h.time)
end

_varidx(s::AthenaKBin, name) = findfirst(==(name), s.var_names)

# Finest block containing (x, y, z) with closed faces; the first block of the highest
# level wins; 0 when none.
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

# Trilinear stencil at one point, shared by every variable read there: owning block `b`
# (0 off the mesh), 1-based lower corner and blend fractions.
struct _BinStencil
    b::Int
    i0::Int; j0::Int; k0::Int
    ti::Float64; tj::Float64; tk::Float64
end

@inline function _point_stencil(s::AthenaKBin, x, y, z)
    b = _locate_block(s, x, y, z)
    b == 0 && return _BinStencil(0, 1, 1, 1, 0.0, 0.0, 0.0)
    dx1 = (s.x1max[b] - s.x1min[b]) / s.nx1
    dx2 = (s.x2max[b] - s.x2min[b]) / s.nx2
    dx3 = (s.x3max[b] - s.x3min[b]) / s.nx3
    fi = (x - s.x1min[b]) / dx1 - 0.5 + 1
    fj = (y - s.x2min[b]) / dx2 - 0.5 + 1
    fk = (z - s.x3min[b]) / dx3 - 0.5 + 1
    i0 = clamp(floor(Int, fi), 1, s.nx1 - 1); ti = clamp(fi - i0, 0.0, 1.0)
    j0 = clamp(floor(Int, fj), 1, s.nx2 - 1); tj = clamp(fj - j0, 0.0, 1.0)
    k0 = clamp(floor(Int, fk), 1, s.nx3 - 1); tk = clamp(fk - k0, 0.0, 1.0)
    return _BinStencil(b, i0, j0, k0, ti, tj, tk)
end

# The blend is `a (1 - t) + b t` per axis, i, then j, then k.
@inline function _sample_stencil(s::AthenaKBin, st::_BinStencil, v::Int)
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

"""
    sample_var(s::AthenaKBin, v, x, y, z) -> Float64

Trilinear sample of variable column `v` (1-based) at `(x, y, z)`, or `NaN` off the
mesh. The owning block is the finest whose closed face box contains the point (the
first in storage order among equals); cell centres sit at `x1min + (i - ½) dx1`, the
lower corner is clamped to `1:nx-1` and the fraction to `[0, 1]`, and each axis blends
as `a (1 - t) + b t`.
"""
function sample_var(s::AthenaKBin, v::Int, x, y, z)
    st = _point_stencil(s, x, y, z)
    st.b == 0 && return NaN
    return _sample_stencil(s, st, v)
end

const _Z4C_CHI_PSI_POWER = -4.0
const _CHI_FLOOR = 1.0e-12

# Columns of the eleven Z4c variables of the ADM reconstruction.
function _z4c_cols(s::AthenaKBin)
    idx(n) = (i = _varidx(s, n); i === nothing && error("missing z4c var $n"); i)
    return (chi = idx("z4c_chi"), gxx = idx("z4c_gxx"), gxy = idx("z4c_gxy"),
            gxz = idx("z4c_gxz"), gyy = idx("z4c_gyy"), gyz = idx("z4c_gyz"),
            gzz = idx("z4c_gzz"), alpha = idx("z4c_alpha"), betax = idx("z4c_betax"),
            betay = idx("z4c_betay"), betaz = idx("z4c_betaz"))
end

"""
    resample_z4c_adm(s::AthenaKBin, half, n; alpha_min = 0.15) -> ADMGrid

The ADM fields of a Z4c dump resampled onto the uniform `n³` node box `[-half, half]³`
(`range(-half, half; length = n)` per axis) as an `ADMGrid` with capture lapse
`alpha_min`. Each node locates its block once and samples the eleven variables with
[`sample_var`](@ref)'s stencil: `α`, `β^i` directly, `γ_ij = ψ⁴ g̃_ij` with
`ψ⁴ = max(χ, 1e-12)^(4/(-4))`. Nodes off the mesh, or where `χ` is `NaN`, get Minkowski
values (`α = 1`, `β = 0`, `γ = δ`). Threaded over `z` slabs.
"""
function resample_z4c_adm(s::AthenaKBin, half::Float64, n::Int; alpha_min::Real = 0.15)
    c = _z4c_cols(s)
    axis = range(-half, half; length = n)
    data = Array{Float64,4}(undef, 10, n, n, n)
    Threads.@threads for k in 1:n
        for j in 1:n, i in 1:n
            _adm_node!(data, s, c, axis, i, j, k)
        end
    end
    return ADMGrid(axis, axis, axis, data; alpha_min = alpha_min)
end

# The ten ADM values (α, β^x, β^y, β^z, γ_xx, γ_xy, γ_xz, γ_yy, γ_yz, γ_zz) at node
# (axis[i], axis[j], axis[k]).
function _adm_node!(data, s, c, axis, i, j, k)
    st = _point_stencil(s, axis[i], axis[j], axis[k])
    chi = st.b == 0 ? NaN : _sample_stencil(s, st, c.chi)
    if isnan(chi)
        data[1, i, j, k] = 1.0
        data[2, i, j, k] = data[3, i, j, k] = data[4, i, j, k] = 0.0
        data[5, i, j, k] = data[8, i, j, k] = data[10, i, j, k] = 1.0
        data[6, i, j, k] = data[7, i, j, k] = data[9, i, j, k] = 0.0
        return nothing
    end
    chi = max(chi, _CHI_FLOOR)
    psi4 = chi^(4.0 / _Z4C_CHI_PSI_POWER)
    data[1, i, j, k] = _sample_stencil(s, st, c.alpha)
    data[2, i, j, k] = _sample_stencil(s, st, c.betax)
    data[3, i, j, k] = _sample_stencil(s, st, c.betay)
    data[4, i, j, k] = _sample_stencil(s, st, c.betaz)
    data[5, i, j, k] = psi4 * _sample_stencil(s, st, c.gxx)
    data[6, i, j, k] = psi4 * _sample_stencil(s, st, c.gxy)
    data[7, i, j, k] = psi4 * _sample_stencil(s, st, c.gxz)
    data[8, i, j, k] = psi4 * _sample_stencil(s, st, c.gyy)
    data[9, i, j, k] = psi4 * _sample_stencil(s, st, c.gyz)
    data[10, i, j, k] = psi4 * _sample_stencil(s, st, c.gzz)
    return nothing
end

"""
    z4c_grid_spacetime(binfile, half, n; alpha_min = 0.15) -> (ADMGrid, AthenaKBin)

Read the Z4c dump `binfile` and return its [`resample_z4c_adm`](@ref) grid on
`[-half, half]³` with `n` nodes per axis, together with the dump. The grid is a frozen
slice (`∂ₜg = 0`), valid while the light-crossing time `≈ 2 half` is short against the
evolution time of the spacetime; a warning says so once per session.
"""
function z4c_grid_spacetime(binfile::AbstractString, half::Float64, n::Int;
                            alpha_min::Real = 0.15)
    @warn "z4c_grid_spacetime: FROZEN-SLICE metric (∂ₜg = 0) from one .bin slice; " *
          "valid only while light-crossing time (~2·half = $(2half)) ≪ the spacetime " *
          "evolution timescale (BBH orbital period). No ∂ₜ(α,β,γ) terms." maxlog=1
    s = read_athenak_bin(binfile)
    return resample_z4c_adm(s, half, n; alpha_min = alpha_min), s
end

"""
    athenak_bin_snapshot(binfile; gamma) -> AthenaKSnapshot

The GRMHD fluid of an AthenaK `.bin` dump (the `mhd_w_bcc` output group) as a
snapshot. `dens, velx, vely, velz, bcc1, bcc2, bcc3` are copied bit for bit;
`eint = press/(gamma - 1)` is computed in `Float64` and rounded once to `Float32`; a
`temperature` column is dropped; non-finite cells are zeroed with one warning. Block
geometry: `dx = (max - min)/n`, `x_left = min + dx/2` per axis; logical locations are
`(min - domain_min)/width`, which must be integral within 1e-6 per block and axis.
"""
function athenak_bin_snapshot(binfile::AbstractString; gamma::Real)
    s = read_athenak_bin(binfile)
    col(n) = (i = _varidx(s, n); i === nothing && error("missing GRMHD var $n in $binfile " *
                                                        "(vars: $(s.var_names))"); i)
    cols = map(col, ("dens", "velx", "vely", "velz", "press", "bcc1", "bcc2", "bcc3"))
    data = _bin_fluid(s, cols, Float64(gamma) - 1.0)
    nmb = s.n_mb
    lo = Float64[v[m] for v in (s.x1min, s.x2min, s.x3min), m in 1:nmb]
    hi = Float64[v[m] for v in (s.x1max, s.x2max, s.x3max), m in 1:nmb]
    ncell = (s.nx1, s.nx2, s.nx3)
    dx = Float64[(hi[a, m] - lo[a, m]) / ncell[a] for a in 1:3, m in 1:nmb]
    xl = Float64[lo[a, m] + dx[a, m] / 2 for a in 1:3, m in 1:nmb]
    dom = (minimum(s.x1min), minimum(s.x2min), minimum(s.x3min))
    ll = Matrix{Int}(undef, 3, nmb)
    for m in 1:nmb, ax in 1:3
        f = (lo[ax, m] - dom[ax]) / (hi[ax, m] - lo[ax, m])
        ll[ax, m] = round(Int, f)
        abs(f - ll[ax, m]) <= 1e-6 ||
            error("athenak_bin_snapshot: block $m axis $ax off the power-of-two " *
                  "logical grid (fractional index $f) — corrupt or unsupported mesh")
    end
    return AthenaKSnapshot(data, lo, hi, xl, dx, s.level, ll; time = s.time)
end

# The feature-fastest `PRIM_NAMES` block of the columns `cols`, press slot divided by `gm1`.
function _bin_fluid(s::AthenaKBin, cols, gm1)
    ni, nj, nk, nmb = s.nx1, s.nx2, s.nx3, s.n_mb
    src = s.data
    data = Array{Float32,5}(undef, 8, ni, nj, nk, nmb)
    nbad = Threads.Atomic{Int}(0)
    Threads.@threads for m in 1:nmb
        bad = 0
        @inbounds for k in 1:nk, j in 1:nj, i in 1:ni, (prim, v) in enumerate(cols)
            val = src[i, j, k, v, m]
            isfinite(val) || (val = 0.0f0; bad += 1)
            data[prim, i, j, k, m] = prim == 5 ? Float32(Float64(val) / gm1) : val
        end
        bad > 0 && Threads.atomic_add!(nbad, bad)
    end
    nbad[] > 0 && @warn "athenak_bin_snapshot: zeroed $(nbad[]) non-finite cell values " *
                        "(isolated c2p failures; zeroed cells emit nothing)" maxlog=1
    return data
end
