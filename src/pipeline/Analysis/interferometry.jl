# Visibility amplitudes of an image along baselines: a Radon line-integral cut at the
# baseline angle, a Tukey window, a zero-padded 1-D FFT and |V| against baseline length
# in Gλ. By the Fourier-slice theorem the transform of the cut is the central slice of
# the 2-D visibility V(u, v) = ∫ I(x, y) exp(−2πi (u x + v y)) dx dy along the baseline;
# a ring of angular diameter d shows minima spaced by ≈ 1/d. Follows AART (MIT, © 2024
# Cárdenas-Avendaño, Zhu & Lupsasca); see NOTICE. Physics: Johnson et al. 2020, Sci.
# Adv. 6, eaaz1310; Cárdenas-Avendaño, Lupsasca & Zhu PRD 107 043030.

"One microarcsecond in radians, `π / 648000 × 10⁻⁶`."
const MUAS_TO_RAD = π / 648000 * 1e-6

"""
    tukey_window(n, α = 0.5) -> Vector{Float64}

The tapered-cosine (Tukey) window of length `n`: one in the centre, with cosine tapers
`w[k+1] = (1 + cos(π (−1 + 2k / (α (n − 1))))) / 2` for `k = 0, …, ⌊α (n − 1) / 2⌋`,
mirrored onto the other end. `α ≤ 0` gives the rectangular window, `α ≥ 1` the Hann
window; `n = 1` gives `[1.0]`. The convention of `scipy.signal.windows.tukey`.
"""
function tukey_window(n::Integer, α::Real = 0.5)
    n == 1 && return ones(Float64, 1)
    α <= 0 && return ones(Float64, n)
    α = min(α, 1.0)
    w = ones(Float64, n)
    width = floor(Int, α * (n - 1) / 2)
    @inbounds for k in 0:width
        w[k+1] = 0.5 * (1 + cos(π * (-1 + 2 * k / (α * (n - 1)))))
        w[n-k] = w[k+1]
    end
    return w
end

# Sample frequencies of an n-point DFT with spacing d, in cycles per unit of d, in FFT
# order: 0, 1, …, ⌈n/2⌉ − 1, then −⌊n/2⌋, …, −1, each times 1/(n d) (numpy `fftfreq`).
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

"""
    radon_cut(img, angle_deg; fov_muas) -> (projection, s_axis)

Parallel-beam Radon projection of the square image `img[iβ, iα]` (rows `β`, columns `α`,
`N × N` pixels over the full field of view `fov_muas` in μas) onto the axis at
`angle_deg` degrees from `+α` toward `+β`. With `c = (N + 1)/2`, the pixel `(iβ, iα)`
sits at offsets `(x, y) = (iα − c, iβ − c)` pixels and projects to `t = x cos θ + y sin θ`;
its value is split linearly between bins `⌊t + c⌋` and `⌊t + c⌋ + 1`, and the parts that
fall outside `1:N` are dropped. Exact for axis-aligned cuts, first order at oblique
angles. Returns the length-`N` projection (sum of pixel values per bin) and its offset
axis `s_axis[k] = (k − c) fov_muas / N` in μas.
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
        x = (iα - c)
        y = (iβ - c)
        t = x * ct + y * st
        f = t + c
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

"""
    visamp(img; angles = (0.0, 90.0), fov_muas, pad = 16, maxbaseline = Inf,
           window = :tukey) -> Vector{NamedTuple}

Normalised visibility amplitude of the square image `img[iβ, iα]` (field of view
`fov_muas` in μas) along each baseline position angle in `angles` (degrees from `+α`
toward `+β`). Per angle: the [`radon_cut`](@ref), times [`tukey_window`](@ref)`(N, 0.5)`
when `window === :tukey` (else no window), zero-padded to `pad × N` samples, FFT, `|V|`.
Baselines are the non-negative FFT frequencies below `maxbaseline`, converted from
cycles per μas to Gλ (10⁹ wavelengths, i.e. cycles per radian / 10⁹); the amplitude is
divided by its value at the first kept baseline (zero) when that is positive. Returns
one `(angle, baselines, amplitude)` per angle, `angle` as `Float64`. A ring of angular
diameter `d` μas shows minima spaced by ≈ `1 / (d MUAS_TO_RAD 10⁹)` Gλ.

# References
Johnson et al. (2020), Sci. Adv. 6, eaaz1310; Cárdenas-Avendaño, Lupsasca & Zhu (2023).
"""
function visamp(img::AbstractMatrix; angles = (0.0, 90.0), fov_muas::Real,
                pad::Integer = 16, maxbaseline::Real = Inf, window::Symbol = :tukey)
    N = size(img, 1)
    Δ = fov_muas / N
    return [_visamp_slice(img, angle, fov_muas, Δ, pad, maxbaseline, window)
            for angle in angles]
end

# One baseline angle of `visamp`; Δ is the pixel size in μas.
function _visamp_slice(img, angle, fov_muas, Δ, pad, maxbaseline, window)
    proj, _ = radon_cut(img, angle; fov_muas = fov_muas)
    w = window === :tukey ? tukey_window(length(proj), 0.5) : ones(length(proj))
    R = proj .* w
    M = pad * length(R)
    Rpad = zeros(Float64, M)
    @inbounds Rpad[1:length(R)] .= R
    amp = abs.(fftshift(fft(Rpad)))
    freqs = fftshift(_fftfreq(M, Δ))         # cycles per μas
    gλ = freqs ./ MUAS_TO_RAD ./ 1e9         # Gλ
    keep = findall(f -> f >= 0 && f < maxbaseline, gλ)
    bl = gλ[keep]
    V = amp[keep]
    V0 = V[1]
    V = V0 > 0 ? V ./ V0 : V
    return (angle = Float64(angle), baselines = bl, amplitude = V)
end

"""
    visamp_minima(baselines, amplitude; skip = 1, bl_min = 0.0) -> Vector{Float64}

Baselines (in the unit of `baselines`, Gλ for [`visamp`](@ref)) of the local minima of
the amplitude curve: interior points `i` with `amplitude[i] < amplitude[i-1]`,
`amplitude[i] ≤ amplitude[i+1]` and `baselines[i] > bl_min`, in order, without the first
`skip` of them (the roll-off of the zero-baseline lobe). When at most `skip` minima are
found they are all returned. Successive spacings measure the ring diameter.
"""
function visamp_minima(baselines::AbstractVector, amplitude::AbstractVector; skip::Integer = 1,
                       bl_min::Real = 0.0)
    mins = Float64[]
    @inbounds for i in 2:length(amplitude)-1
        if amplitude[i] < amplitude[i-1] && amplitude[i] <= amplitude[i+1] && baselines[i] > bl_min
            push!(mins, baselines[i])
        end
    end
    return length(mins) > skip ? mins[skip+1:end] : mins
end
