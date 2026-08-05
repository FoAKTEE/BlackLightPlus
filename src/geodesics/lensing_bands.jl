# Geodesics — lensing_bands.jl
# Host-side setup for photon-ring imaging: the Kerr critical curve (shadow
# boundary), the apparent-horizon (n=0 inner) boundary, the n-band inner/outer
# boundaries, and per-band adaptive Cartesian (α,β) grids + masks — the
# once-per-image geometry deciding WHICH pixels belong to photon-ring layer n
# (per-pixel tracing is `analytic_emission`, analytic_kerr.jl). Ported from AART
# (MIT, © 2024 Cárdenas-Avendaño, Zhu & Lupsasca); notices preserved per their
# terms. Physics: Gralla & Lupsasca PRD 101 044032 (P1)/044033 (P2); Cárdenas-
# Avendaño, Lupsasca & Zhu PRD 107 043030 (P3); eq. tags (Pn Eq. m) cite these.

# ── small numerical helpers ──────────────────────────────────────────────────

"Signed polygon area (shoelace); |area| is the enclosed area for a simple loop."
function polygon_area(xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    s = zero(eltype(xs))
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        s += xs[i] * ys[j] - xs[j] * ys[i]
    end
    return abs(s) / 2
end
polygon_area(poly::AbstractMatrix) = polygon_area(view(poly, :, 1), view(poly, :, 2))

"Winding-number point-in-polygon (true if (px,py) strictly inside the closed loop)."
function _in_polygon(px, py, xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    wn = 0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        yi = ys[i]; yj = ys[j]
        if yi <= py
            if yj > py && ((xs[j] - xs[i]) * (py - yi) - (px - xs[i]) * (yj - yi)) > 0
                wn += 1
            end
        else
            if yj <= py && ((xs[j] - xs[i]) * (py - yi) - (px - xs[i]) * (yj - yi)) < 0
                wn -= 1
            end
        end
    end
    return wn != 0
end

# Row bucket for `band_grid`: the edges of the closed loop (xs, ys) crossing the
# horizontal line y = py, under `_in_polygon`'s exact half-open rule (yi ≤ py < yj
# upward, yj ≤ py < yi downward). Only these edges can contribute to the winding
# number of any point on the row, so the per-point test below stays BIT-IDENTICAL
# to `_in_polygon` while skipping the O(P) non-crossing edges.
function _row_edges!(buf::Vector{NTuple{4,Float64}}, xs::AbstractVector,
                     ys::AbstractVector, py)
    empty!(buf)
    n = length(xs)
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        yi = ys[i]; yj = ys[j]
        if (yi <= py) != (yj <= py)
            push!(buf, (xs[i], yi, xs[j], yj))
        end
    end
    return buf
end

# `_in_polygon` restricted to a row's crossing edges (same cross-product test,
# same ±1 accumulation — the winding number is an exact integer either way).
@inline function _wn_row(px, py, edges::Vector{NTuple{4,Float64}})
    wn = 0
    @inbounds for (xi, yi, xj, yj) in edges
        if yi <= py
            ((xj - xi) * (py - yi) - (px - xi) * (yj - yi)) > 0 && (wn += 1)
        else
            ((xj - xi) * (py - yi) - (px - xi) * (yj - yi)) < 0 && (wn -= 1)
        end
    end
    return wn != 0
end

"Bisection root of monotone `f` bracketed by a sign change on [lo,hi]; NaN if none."
function _bisect(f, lo::T, hi::T; iters=80, nscan=64) where {T}
    flo = f(lo)
    a = lo; fa = flo
    b = hi; fb = flo
    found = false
    # scan for a sign change (boundaries are monotone but the valid bracket is unknown)
    for i in 1:nscan
        x = lo + (hi - lo) * i / nscan
        fx = f(x)
        if isfinite(fx) && isfinite(fa) && fa * fx <= 0
            b = x; fb = fx; found = true; break
        end
        a = x; fa = fx
    end
    found || return T(NaN)
    for _ in 1:iters
        m = (a + b) / 2
        fm = f(m)
        if !isfinite(fm)
            b = m; continue
        end
        (fa * fm <= 0) ? (b = m; fb = fm) : (a = m; fa = fm)
    end
    return (a + b) / 2
end

# The radial potential R(r) (P2 Eq. 5) is `kerr_radial_potential` (kerr_orbits.jl);
# its reciprocal square root is the Mino-time integrand ∫ dr/√R (P1 Eq. 8a).

# ∫_{rP}^{ro} dr/√R via the r = 1/x substitution (smooths the 1/r² tail) +
# composite Simpson. Used by the apparent-horizon boundary condition.
function _radial_mino_horizon_to_obs(a, λ, η, rP, ro; N=4000)
    xlo = 1 / ro; xhi = 1 / rP
    h = (xhi - xlo) / N
    acc = 0.0
    @inbounds for i in 0:N
        x = xlo + i * h
        r = 1 / x
        Rr = kerr_radial_potential(r, a, λ, η)
        val = Rr > 0 ? (1 / (x * x)) / sqrt(Rr) : 0.0   # dr = dx/x²
        w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        acc += w * val
    end
    return acc * h / 3
end

# ── critical curve (Kerr shadow boundary) ────────────────────────────────────

# Photon-shell (λ, η) parametrized by the shell radius r (P2 Eqs. 38–39): the
# critical-curve map. `(r² − 2r + a²)` is Δ(r) = `kerr_delta`.
@inline function _shell_lambda_eta(r, a)
    Δ = kerr_delta(r, a)
    λ = a + r / a * (r - 2 * Δ / (r - 1))
    η = r^3 / a^2 * (4 * Δ / (r - 1)^2 - r)
    return λ, η
end

"""
    critical_curve(a, θo; n=100) -> (α, β)

The Kerr critical curve (shadow boundary) on the Bardeen screen for spin `a`,
observer inclination `θo` [rad] (P2 Eqs. 38–39, 55), as a closed parametric loop
of `≈2n` points (upper branch + mirrored lower branch). For a → 0 it is the
Schwarzschild circle of radius 3√3 M, returned analytically (machine precision).
"""
function critical_curve(a::Float64, θo::Float64; n::Int=100)
    R = 3 * sqrt(3.0)
    if abs(a) < 1e-8                                   # Schwarzschild: exact circle
        ψ = range(0, 2π; length=2n)
        return (R .* cos.(ψ), R .* sin.(ψ))
    end
    rM = photon_orbit_radius(a; prograde=true)         # P2 Eq. 40 (photon-shell bounds)
    rP = photon_orbit_radius(a; prograde=false)
    st = sin(θo); ct = cos(θo)
    cot2 = (ct / st)^2
    β2(r) = begin
        λ, η = _shell_lambda_eta(r, a)
        η + a^2 * ct^2 - λ^2 * cot2
    end
    # refine the r-range where β² ≥ 0
    rs = range(rM, rP; length=200000)
    lo = rM; hi = rP; seen = false
    for r in rs
        if β2(r) > 0
            seen || (lo = r; seen = true)
            hi = r
        end
    end
    seen || return (Float64[], Float64[])
    lo += 1e-12; hi -= 1e-12
    αu = Float64[]; βu = Float64[]
    for r in range(lo, hi; length=n)
        λ, η = _shell_lambda_eta(r, a)
        b2 = η + a^2 * ct^2 - λ^2 * cot2
        push!(αu, -λ / st); push!(βu, sqrt(max(b2, 0.0)))
    end
    α = vcat(αu, reverse(αu))
    β = vcat(βu, -reverse(βu))
    return (α, β)
end

# ── apparent horizon (n=0 inner boundary) ────────────────────────────────────

# AART ApparentBH(s,...) boundary residual: a ray whose screen point is s×(α0,β0)
# originates exactly at the horizon (P1 Eqs. 8a, 20).
function _apparent_bh_residual(s, a, θo, α0, β0, m, sgn, ro)
    α = s * α0; β = s * β0
    λ, η = conserved_quantities(α, β, θo, a)
    uP, uM = angular_turning_points(λ, η, a)
    rP = kerr_horizon(a)
    (uP <= 0 || uM >= 0) && return NaN
    I = _radial_mino_horizon_to_obs(a, λ, η, rP, ro)
    return sqrt(-uM * a^2) * I +
           sgn * JacobiElliptic.F(asin(clamp(cos(θo) / sqrt(uP), -1.0, 1.0)), uP / uM) -
           2 * m * JacobiElliptic.K(uP / uM)
end

"""
    apparent_horizon(a, θo; n=100, ro=1e5) -> (α, β)

The apparent-horizon curve — the n=0 lensing band's INNER boundary (image of the
event horizon). For each critical-curve direction the screen point is scaled by
`s ∈ (0,1)` until a ray there originates exactly at the horizon (AART `ApparentBH`
root). For a → 0 the curve is a circle (Schwarzschild); with spin it is D-shaped.
"""
function apparent_horizon(a::Float64, θo::Float64; n::Int=100, ro::Float64=1e5)
    if cos(θo) < 0
        # θo > π/2: equatorial mirror θo → π−θo ⇒ β → −β — the same normalisation
        # `analytic_emission` applies. (The critical curve is β-symmetric, so it
        # needs no mirror; this curve's m=1/m=0 legs make it asymmetric.)
        αm, βm = apparent_horizon(a, π - θo; n=n, ro=ro)
        return (αm, -βm)
    end
    αc, βc = critical_curve(a, θo; n=n)
    isempty(αc) && return (Float64[], Float64[])
    αo = similar(αc); βo = similar(βc)
    for i in eachindex(αc)
        α0 = αc[i]; β0 = βc[i]
        m, sgn = β0 >= 0 ? (1.0, 1.0) : (0.0, -1.0)
        f = s -> _apparent_bh_residual(s, a, θo, α0, β0, m, sgn, ro)
        s = _bisect(f, 0.05, 0.999)
        s = isnan(s) ? 0.99 : s
        αo[i] = s * α0; βo[i] = s * β0
    end
    return (αo, βo)
end

# ── n-band boundaries (AART nlayers) ─────────────────────────────────────────

# nlayers residual for equatorial disk (θd = π/2 ⇒ the disk-source term vanishes).
# outer=true → the s>1 (Type II) branch; outer=false → the s<1 (Type III) branch.
function _nlayers_residual(s, a, θo, α0, β0, mbar, outer::Bool)
    α = s * α0; β = s * β0
    m = mbar + (β0 >= 0 ? 1 : 0)                       # H(β0): equator-crossing count,
                                                       # H(0)=1 (matches analytic_emission)
    λ, η = conserved_quantities(α, β, θo, a)
    η < 0 && return NaN
    νθ = β0 >= 0 ? 1.0 : -1.0                          # H(0)=1 pairing (sign(±0.0)
                                                       # would decouple νθ from m)
    r1, r2, r3, r4 = radial_roots(λ, η, a)
    uP, uM = angular_turning_points(λ, η, a)
    (uP <= 0 || uM >= 0) && return NaN
    kang = uP / uM
    Gθ = 1 / (sqrt(-uM) * a) *
         (2 * m * JacobiElliptic.K(kang) -
          νθ * JacobiElliptic.F(asin(clamp(cos(θo) / sqrt(uP), -1.0, 1.0)), kang))
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r21 = r2 - r1
    if outer                                           # Type II: P2 Eq. A10
        kk = real(r32 * r41 / (r31 * r42))
        Q1 = 4 / real(sqrt(r31 * r42)) *
             JacobiElliptic.F(asin(clamp(real(sqrt(r31 / r41)), -1.0, 1.0)), kk)
        return Q1 - Gθ
    else                                               # Type III: P2 Eq. A11
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        k3 = ((A + B)^2 - real(r21)^2) / (4 * A * B)
        k3 >= 1 && return NaN
        rP = kerr_horizon(a)
        xc = clamp((A - B) / (A + B), -1.0, 1.0)
        xh = clamp(real((A * (rP - r1) - B * (rP - r2)) / (A * (rP - r1) + B * (rP - r2))), -1.0, 1.0)
        Q2 = 1 / sqrt(A * B) * (JacobiElliptic.F(acos(xc), k3) - JacobiElliptic.F(acos(xh), k3))
        return Q2 - Gθ
    end
end

"""
    lensing_band(a, θo, n; npts=100, box=25.0) -> (inner, outer)

Inner and outer boundary curves (each an `N×2` matrix of (α,β)) of lensing band
`n` for spin `a`, inclination `θo` [rad] (θo > π/2 maps through the equatorial
mirror β → −β, as in `analytic_emission`). Band n=0 is bounded by the apparent
horizon (inner) and the image box of half-width `box` (outer). Bands n ≥ 1 are
thin annuli straddling the critical curve: for each critical-curve direction the
inner edge scales inward (s<1, Type III root of AART `nlayers`) and the outer
edge outward (s>1, Type II root), both hand-rolled bisections.
"""
function lensing_band(a::Float64, θo::Float64, n::Int; npts::Int=100, box::Float64=25.0)
    if cos(θo) < 0
        # θo > π/2: equatorial mirror θo → π−θo ⇒ β → −β, matching the
        # `analytic_emission` convention ((α,β) ∈ band(θo) ⇔ (α,−β) ∈ band(π−θo)).
        inner, outer = lensing_band(a, π - θo, n; npts=npts, box=box)
        inner[:, 2] .= .-inner[:, 2]
        outer[:, 2] .= .-outer[:, 2]
        return inner, outer
    end
    αc, βc = critical_curve(a, θo; n=npts)
    if n == 0
        αi, βi = apparent_horizon(a, θo; n=npts)
        inner = hcat(αi, βi)
        outer = [ box  box; -box  box; -box -box;  box -box]
        return inner, outer
    end
    αi = similar(αc); βi = similar(βc); αo = similar(αc); βo = similar(βc)
    for i in eachindex(αc)
        α0 = αc[i]; β0 = βc[i]
        s_in = _bisect(s -> _nlayers_residual(s, a, θo, α0, β0, n, false), 0.2, 0.99999)
        s_out = _bisect(s -> _nlayers_residual(s, a, θo, α0, β0, n, true), 1.00001, 4.0)
        lo = 4.0                       # high inclination pushes the outer edge past
        while isnan(s_out) && lo < 256.0   # s=4 — grow the bracket geometrically
            s_out = _bisect(s -> _nlayers_residual(s, a, θo, α0, β0, n, true), lo, 2 * lo)
            lo *= 2
        end
        isnan(s_in) &&
            @warn "lensing_band: inner-edge root not bracketed; using fallback s=0.999" a θo n maxlog = 1
        isnan(s_out) &&
            @warn "lensing_band: outer-edge root not bracketed; using fallback s=1.001" a θo n maxlog = 1
        s_in = isnan(s_in) ? 0.999 : s_in
        s_out = isnan(s_out) ? 1.001 : s_out
        αi[i] = s_in * α0; βi[i] = s_in * β0
        αo[i] = s_out * α0; βo[i] = s_out * β0
    end
    return hcat(αi, βi), hcat(αo, βo)
end

# ── per-band Cartesian grid + mask ───────────────────────────────────────────

"""
    band_grid(inner, outer; dx=0.02, box=nothing) -> (grid, mask, N)

Adaptive Cartesian (α,β) grid covering the outer boundary's bounding box at
spacing `dx`, with `mask[i]` true where grid point i lies inside `outer` but
outside `inner` — i.e. in the lensing band (AART `grid_mask`, winding-number
point-in-polygon). `grid` is `M×2`; `N` is the per-side grid count.
"""
function band_grid(inner::AbstractMatrix, outer::AbstractMatrix; dx::Float64=0.02, box=nothing)
    oxs = view(outer, :, 1); oys = view(outer, :, 2)
    lim = box === nothing ? maximum(abs, outer) + 5dx : box
    N = 2 * cld(round(Int, lim / dx), 1)
    xs = range(-lim, lim; length=N)
    ixs = view(inner, :, 1); iys = view(inner, :, 2)
    test_inner = size(inner, 1) >= 3
    npts = N * N
    grid = Matrix{Float64}(undef, npts, 2)
    mask = Vector{Bool}(undef, npts)
    # Per-ROW winding test: bucket each polygon's row-crossing edges once (a few
    # of the ~npts vertices), then test the N points of the row against only
    # those — O(N·(P + k·N)) total instead of the brute-force O(N²·P), with a
    # bit-identical mask (same per-edge arithmetic; see `_row_edges!`).
    oedges = NTuple{4,Float64}[]
    iedges = NTuple{4,Float64}[]
    idx = 0
    for yy in xs
        _row_edges!(oedges, oxs, oys, yy)
        test_inner && _row_edges!(iedges, ixs, iys, yy)
        for xx in xs
            idx += 1
            grid[idx, 1] = xx; grid[idx, 2] = yy
            inside_out = _wn_row(xx, yy, oedges)
            mask[idx] = inside_out && !(test_inner && _wn_row(xx, yy, iedges))
        end
    end
    return grid, mask, N
end
