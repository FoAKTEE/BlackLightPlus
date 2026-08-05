# interferometry.jl — image → visibility-amplitude pipeline: a Radon
# line-integral cut of the image at a baseline angle → Tukey window → zero-padded
# 1-D FFT → |V|(baseline) in Gλ. By the Fourier-slice theorem this is the central
# slice of the 2-D image visibility along that baseline; the photon-ring diameter
# d appears as a ≈ 1/d spacing of the visibility-amplitude minima.
# Ported from AART (MIT, © 2024 Cárdenas-Avendaño, Zhu & Lupsasca;
# github.com/iAART/aart, aart_func/visamp_f.py); MIT notices preserved.
# Physics: Johnson et al. 2020, Sci. Adv. 6, eaaz1310; Cárdenas-Avendaño,
# Lupsasca & Zhu, PRD 107, 043030. Self-contained except FFTW.

using FFTW: fft, fftshift

"One microarcsecond in radians (AART `muas_to_rad`): `π / (180·3600·10⁶)`. Converts a μas image scale / baseline to radians and back."
const MUAS_TO_RAD = π / 648000 * 1e-6

# ── Tukey (tapered-cosine) window — scipy.signal.windows.tukey parity ────────

"""
    tukey_window(n, α=0.5) -> Vector

The tapered-cosine (Tukey) window of length `n`: a flat centre with cosine
tapers over the outer `α` fraction (α=0 rectangular, α=1 Hann). Matches
`scipy.signal.windows.tukey`; AART uses α=0.5 to suppress FFT ringing from the
finite Radon cut without over-smoothing the ring minima.
"""
function tukey_window(n::Integer, α::Real=0.5)
    n == 1 && return ones(Float64, 1)
    α <= 0 && return ones(Float64, n)
    α = min(α, 1.0)
    w = ones(Float64, n)
    width = floor(Int, α * (n - 1) / 2)
    @inbounds for k in 0:width
        w[k+1] = 0.5 * (1 + cos(π * (-1 + 2 * k / (α * (n - 1)))))
        w[n-k] = w[k+1]                       # symmetric taper
    end
    return w
end

# numpy.fft.fftfreq(n, d): sample frequencies in cycles per unit of `d`.
function _fftfreq(n::Integer, d::Real)
    val = 1 / (n * d)
    f = Vector{Float64}(undef, n)
    nn = (n - 1) ÷ 2 + 1
    @inbounds for k in 0:nn-1
        f[k+1] = k * val
    end
    @inbounds for k in nn:n-1
        f[k+1] = (k - n) * val
    end
    return f
end

# ── Radon line-integral cut ──────────────────────────────────────────────────

"""
    radon_cut(img, angle_deg; fov_muas) -> (projection, s_axis_muas)

Parallel-beam Radon projection of a square image (`I[iβ, iα]`, rows = β) at
baseline `angle_deg`; `fov_muas` is the full field of view in μas. Each pixel is
linearly splatted into the two bins straddling its projected coordinate
`t = x·cosθ + y·sinθ` — exact for axis-aligned cuts, first-order at oblique
angles (sufficient for the visibility-amplitude ring ruler). Returns the
length-N projection and its μas offset axis.
"""
function radon_cut(img::AbstractMatrix, angle_deg::Real; fov_muas::Real)
    N = size(img, 1)
    @assert size(img, 2) == N "radon_cut expects a square image"
    Δ = fov_muas / N
    c = (N + 1) / 2
    θ = deg2rad(angle_deg)
    ct = cos(θ); st = sin(θ)
    proj = zeros(Float64, N)
    @inbounds for iβ in 1:N, iα in 1:N
        v = img[iβ, iα]
        v == 0 && continue
        x = (iα - c)          # α offset (pixels)
        y = (iβ - c)          # β offset (pixels)
        t = x * ct + y * st   # projected coordinate (pixels)
        f = t + c             # fractional bin index
        b0 = floor(Int, f)
        wgt = f - b0
        if 1 <= b0 <= N
            proj[b0] += v * (1 - wgt)
        end
        if 1 <= b0 + 1 <= N
            proj[b0+1] += v * wgt
        end
    end
    s_axis = [(k - c) * Δ for k in 1:N]
    return proj, s_axis
end

# ── visibility amplitude ─────────────────────────────────────────────────────

"""
    visamp(img; angles=(0.0, 90.0), fov_muas, pad=16, maxbaseline=Inf,
           window=:tukey) -> Vector{NamedTuple}

Interferometric visibility amplitude of `img` along each baseline `angle`
(degrees): `radon_cut` → Tukey window (`window=:tukey`, else rectangular) →
zero-pad by `pad`× → 1-D FFT → `|V|` (normalised to `|V(0)| = 1`) on the
non-negative baselines below `maxbaseline`, in Gλ. Returns one
`(angle, baselines, amplitude)` entry per angle. The minima are spaced by
≈ `1/(d · MUAS_TO_RAD · 1e9)` Gλ for a ring of angular diameter `d` μas.
"""
function visamp(img::AbstractMatrix; angles=(0.0, 90.0), fov_muas::Real,
                pad::Integer=16, maxbaseline::Real=Inf, window::Symbol=:tukey)
    N = size(img, 1)
    Δ = fov_muas / N                     # pixel angular size along the projection (μas)
    out = NamedTuple[]
    for angle in angles
        proj, _ = radon_cut(img, angle; fov_muas=fov_muas)
        w = window === :tukey ? tukey_window(length(proj), 0.5) : ones(length(proj))
        R = proj .* w
        M = pad * length(R)
        Rpad = zeros(Float64, M)
        @inbounds Rpad[1:length(R)] .= R
        amp = abs.(fftshift(fft(Rpad)))
        freqs = fftshift(_fftfreq(M, Δ))         # cycles / μas
        gλ = freqs ./ MUAS_TO_RAD ./ 1e9         # baseline in Gλ
        keep = findall(f -> f >= 0 && f < maxbaseline, gλ)
        bl = gλ[keep]
        V = amp[keep]
        V0 = V[1]                                 # |V| at the (near-)zero baseline
        V = V0 > 0 ? V ./ V0 : V
        push!(out, (angle=Float64(angle), baselines=bl, amplitude=V))
    end
    return out
end

"""
    visamp_minima(baselines, amplitude; skip=1, bl_min=0.0) -> Vector

Baselines (Gλ) of the local minima of a visibility-amplitude curve — the
strict-interior valleys with `bl > bl_min`, dropping the first `skip` (the DC
lobe roll-off). Successive minima spacings give the photon-ring-diameter ruler.
"""
function visamp_minima(baselines::AbstractVector, amplitude::AbstractVector; skip::Integer=1, bl_min::Real=0.0)
    mins = Float64[]
    @inbounds for i in 2:length(amplitude)-1
        if amplitude[i] < amplitude[i-1] && amplitude[i] <= amplitude[i+1] && baselines[i] > bl_min
            push!(mins, baselines[i])
        end
    end
    return length(mins) > skip ? mins[skip+1:end] : mins
end
