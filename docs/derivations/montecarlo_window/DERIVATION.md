# Monte Carlo transport through an evolving plasma: derivation

The stationary Monte Carlo module (`mc_spectrum` without `window`) emits every superphoton at
coordinate time 0 from one frozen plasma state and reports νL_ν per unit coordinate time.
With `window = (t_a, t_b)` the plasma is a function of time, every superphoton is emitted at
its own time t_e in the window, the march reads plasma and metric at the photon's coordinate
time x⁰, and escaped photons are binned in the arrival time at a distant observer. This note
derives every formula the code uses and the closed forms of the exact tests, and records
where the proposal (DESIGN.md) is corrected.

Conventions: signature (−,+,+,+), G = c = 1, Cartesian coordinates x^μ = (t, x, y, z) in units
of L = L_unit (GM/c² on a grid, 1 cm for a sphere), time unit T = L/c. Photon momenta are
covariant, p_μ, in mₑc²; E = −p_t.

## 1. The emitted number and the zone measure

**Counting.** In the fluid frame a thermal plasma emits j_ν/(hν) photons per unit volume,
time, frequency and solid angle (all fluid-frame quantities). The number of emission events in
a spacetime region is a scalar, and the fluid-frame volume element dV_f dτ_f is the invariant
4-volume √−g d⁴x. Hence

    dN = √−g dt d³x ∫dν ∫dΩ j_ν/(hν),        dṄ/d ln ν = (√−g Δ³x L³ / h) ∫ j_ν dΩ        (1)

photons per unit *coordinate* time from a cell of coordinate volume Δ³x, with ∫ j_ν dΩ at the
fluid-frame ν. This is the measure of the stationary module, dV = √−g Δx¹Δx²Δx³ L³ (`ZoneSite.dV`).

*Correction to DESIGN §1.* The proposal writes √−g (u^t). The fluid-frame volume of the cell's
fluid is dV_f = u^t √−g d³x and its proper time dτ_f = dt/u^t; the u^t cancels in the rate per
coordinate time. A factor u^t would count photons per unit fluid proper time.

**Angle-integrated emissivity.** With X = ν/ν_s, ν_s = (2/9)ν_c Θₑ² sin θ, the fit depends on ν,
Θₑ only through K = ν/((2/9)ν_c Θₑ²), so ∫j_ν dΩ = 4π (n_e e²ν_c/c)(√2π/27) F(K) with the one-
variable integral F(K) = ∫₀^{π/2} sin²θ G(K/sin θ) dθ, tabulated once (`EmissivityTable`). It
enters both the weight table and every zone's count, so the frequency acceptance and its
envelope are the same function.

**Weights and counts, stationary.** On n + 1 log-spaced nodes ν_m over [ν_min, ν_max] with
spacing Δ = ln(ν_max/ν_min)/n and the trapezoid weights c_0 = c_n = 1/2, c_m = 1 inside,

    w_m = Σ_z ∫j dΩ(ν_m) dV_z / (h N_s'),    N_s' = N_s / ln(ν_max/ν_min),
    N_z = (dV_z/h) Δ Σ_m c_m ∫j dΩ(ν_m)/w_m,                Σ_z N_z = N_s' Δ Σ_m c_m = N_s.  (2)

Each zone's N_z is rounded stochastically (⌊N_z⌋ + Bernoulli(N_z − ⌊N_z⌋), zone order, stream
MCRng(seed, 0)), so the expected number of primaries is N_s. A primary of zone z draws ln ν
uniformly and accepts with (∫j dΩ/w)/max_m(∫j dΩ/w) and carries weight w(ν) (photons per unit
time). Its expected weight per ln ν is then (dV_z/h)∫j dΩ times the ratio of the trapezoid to
the exact ln ν integral of ∫jdΩ/w, which differs from 1 by O(Δ²) (the log-linear interpolation of
w between nodes). Stride-1 zones are every owned cell above the horizon proxy; a stride s draws
one owned cell per s³ group on a seed stream and gives it n_own cells' volume, an unbiased
quadrature of the stride-1 sum.

## 2. Emission over a window

**Rate convention.** A packet of weight w stands for w (t_b − t_a) T physical photons: w is the
packet's share of the *window mean* of the emission rate. With this convention the weight table
and the counts (2) are built from the window-mean emission, every spectrum of the run is the
window mean of the stationary estimator, and the fluence is (t_b − t_a) T times it.

**Emission nodes.** The emission rate density in (t, zone, ν, direction) is written as a sum of
nodes,

    ṅ(t, ·) = Σ_m φ_m(t) ṅ_m(·),                                                       (3)

where ṅ_m is the complete stationary emission (counts, spectrum, angular pattern, kinematics) of
one plasma state, placed at the time t, and φ_m ≥ 0 a profile in time. The kinds of matter:

| matter | nodes ṅ_m | profiles φ_m | placement of a node photon at t |
|---|---|---|---|
| a snapshot (frozen) | the snapshot | 1 on [t_a, t_b] | time translation |
| `Corotating(snap, ω)` | the snapshot at t = 0 | 1 on [t_a, t_b] | time translation and R_z(ω t) |
| `FrameSequence` of frames F_k at τ_k | runs of equal frames | sum of the run's hats | time translation |
| `VariableSphere` with factors f_k at τ_k | the sphere at n_e f_k | sum of the run's hats | time translation |

The hat of node k is the piecewise-linear function with φ_k(τ_k) = 1, φ_k(τ_{k±1}) = 0 (only its
inner half at the series ends), so Σ_k φ_k ≡ 1 on [τ_1, τ_K]: (3) is the linear interpolation in
time of the frames' emission. A run of consecutive equal frames (the same object, on a
stationary spacetime; equal factors of a variable sphere) is one node whose profile is the sum
of its hats, flat on the run and ramped to its neighbours; the profile is clipped to the window
and needs no knots inside a flat stretch, so a run covering the window has the two knots
(t_a, 1), (t_b, 1).

*Correction to DESIGN §1.* The proposal draws t_e from P(t) "piecewise linear because the
primitives are interpolated linearly". The emissivity of linearly blended primitives is not linear
in t (n_e, Θₑ and B enter nonlinearly through the decode and the synchrotron fit). It is the
interpolation (3) of the frames' emission that is piecewise linear, and that is the emission
model; the transport blends the primitives (section 3). The two interpolants of the same frames
differ at second order in the frame spacing, the counterpart of the cell-centre emission against
the interpolated absorption in space. For a variable sphere the emission and the absorptivity are
both proportional to n_e and n_e(t) is the piecewise-linear interpolant itself, so emission model
and transport coincide exactly.

**Shares and the zone measure.** The node's share of the window's emission is

    W_m = ∫_{t_a}^{t_b} φ_m dt / (t_b − t_a),         Σ_m W_m = 1 when Σ φ_m ≡ 1 on the window. (4)

The window mean of (3) is Σ_m W_m ṅ_m, a stationary emission whose zones are the nodes' zones
with the measure W_m dV. The code therefore concatenates the nodes' zone lists with volumes
scaled by W_m and builds the one weight table and the counts (2) from that list: the expected
number of primaries is N_s, the weight table is that of the window mean, and every packet of
every node carries w(ν) of one table. The profile integrals are trapezoid sums of the knots,
exact for piecewise-linear φ.

**Sampling a primary.** Primary g belongs to zone z of node m by the cumulative counts. Its
emission time is drawn from φ_m: a segment of the knots by its share of the area (one uniform;
none for one segment), then on the segment the density ∝ (1 − s)v₀ + s v₁, s ∈ [0, 1], as the
mixture of 2(1 − s) (weight v₀) and 2s (weight v₁): with probability v₀/(v₀ + v₁) s = 1 − √(1 − u),
else s = √u (one uniform for the branch, one for s; a flat segment takes s = u). This is exact:
∫₀^x 2(1 − s)ds = 1 − (1 − x)² and ∫₀^x 2s ds = x². The joint density of (m, z, t) is
φ_m(t) N_{z,m}, which is (3) with the counts as the zone measure; drawing t first and then the node
with probability ∝ φ_m(t) (the proposal) has the same joint density, without the stratification by
node and zone.

**Streams.** The emission time draws from the primary's own Philox identity at draw index 2³¹
(`time_stream`), the stationary generation and march from index 0; a march draws far fewer than
2³¹ times, so the photon's stationary draws are untouched and the emission time is independent of
them.

**Placement.** Every node photon is generated exactly as by the stationary module (the zone of the
node at coordinate time 0, its kinematics and metric there) and then carried to its emission time:

- *time translation* x⁰ → t_e. On a stationary spacetime with a node that does not depend on time
  the translated photon is the photon emitted at t_e: the metric, the momentum and the zone are
  unchanged. A frozen snapshot or a series therefore emits over a window only on a stationary
  spacetime (`helical_symmetry(st, 0)`); elsewhere the code refuses.
- *helical map* for a plasma co-rotating at ω: (x, y) → R_z(ω t_e)(x, y), and the spatial covariant
  momentum p_i → R_z(ω t_e) p_i (rotations are orthogonal, so covariant and contravariant spatial
  components transform alike), E = −p_t unchanged. If the spacetime has the helical Killing vector
  ∂_t + ω∂_φ, g_μν(R x, t + τ) = R g(x, t) Rᵀ, the photon is null in the metric at its emission event
  and its zone count is the reference zone's: √−g, the decoded n_e, Θₑ, b² and the horizon proxy
  are scalars of the isometry. The code requires `helical_symmetry(st, ω)`: Minkowski and Kerr–Schild
  Kerr (spin along z) at every ω, `BoostedSuperposedKerrSchild` on a `CircularOrbit` of frequency ω
  with spins along z (s_A(t + τ) = R s_A(t), v_A likewise, and each boosted Kerr–Schild term is built
  from dot and cross products of x − s_A, v_A and a_A), and any stationary spacetime at ω = 0.

## 3. Transport at the photon's time

The march (`track!`) evaluates the metric at the photon's state (x⁰ included) at every step, so a
time-dependent metric is read at the photon's time without change. The plasma is a provider read at
the sample's x⁰:

- `FrameSequence`: frames at times τ_k, bracket of x⁰, samples of the two frames blended linearly in
  the primitives (`blend`), the frame itself where the two are the same object or the weight is 0 or
  1, clamped to the end frames outside [τ_1, τ_K] (the convention of slow-light imaging).
- `Corotating(P, ω)`: the primitives of P at R_z(−ω x⁰)x with velocity and field rotated by R_z(ω x⁰).
- `VariableSphere`: n_e(x⁰) from the piecewise-linear factor.

A scattered child starts at its event with the event's x⁰, so it inherits the parent's clock. The
time of a march spent after the last frame is recorded, weighted by the photon's starting weight
(`beyond_series`): the clamped frame stands in for the plasma there.

## 4. Arrival time at a distant observer

A photon escapes at the event (t_e, x_e) on the escape sphere (the step after which r > R_esc, or
r > R_em outbound) with spatial direction n̂ = k⃗/|k⃗|, k^i = g^{iν}p_ν. In flat space it moves on
x(t) = x_e + n̂(t − t_e) and crosses the plane n̂·x = D, the wavefront of a distant observer along n̂
at distance D, at

    t = t_e + D − n̂·x_e,       so      t_arr ≡ t_e − n̂·x_e                                (5)

is its arrival time up to the constant D (the arrival time of light leaving the origin at t = 0).
Along the straight line t_e − n̂·x_e is constant, so (5) does not depend on where the photon is
stopped. Each photon is assigned to the observer along its own direction; a band of solid angle
ΔΩ averages the light curves of the observers in it.

Beyond the stop the spacetime is that of the total mass M at r ≫ M (b the impact parameter).
In ingoing Kerr–Schild coordinates an outgoing radial null ray has dr/dt = (r − 2M)/(r + 2M), so
t − r − 4M ln(r − 2M) is constant along it, and the coordinate time from r_e to D exceeds the flat
value by 4M ln(D/r_e) + O(M²/r_e) (2M ln in Schwarzschild or harmonic time). Hence

    t_obs = D + 4M ln D + t_arr − 4M ln r_e + O(M b²/r_e², M²/r_e).                       (6)

*Correction to DESIGN §3.* The dropped constant is D + 4M ln D in Kerr–Schild time, not
D + 2M ln(D/R_esc). What remains photon-dependent is −4M ln r_e and the impact-parameter terms. On a
grid most photons stop outbound at the emission radius R_em (at least 100 M and 1.2 times the
outermost zone), the rest at R_esc = 10³ M; either stop overshoots its radius by at most one step,
Δr ≈ r/C, so −4M ln r_e varies by at most 4M ln(1 + 1/C) ≈ 4M/C (0.05 M at C = 80, M = 1) among the
photons of one stop and by 4M ln(R_esc/R_em) ≈ 9 M between the two stops, a fixed offset of the
photons that leave inbound through R_em (rare: they pass the source again). The bending left beyond
the stop changes n̂ by about 2Mb/r_e² and n̂·x_e by about 2Mb²/r_e², at most of order 2M for
b ≲ r_e. All of these are far below the arrival bins of an orbit (P/16 ≈ 49 M for the binary of
separation 25 M). The code uses (5); on a flat spacetime it is exact.

## 5. The estimator

**Per-bin luminosity.** A packet of weight w carries w(t_b − t_a)T photons. The energy reaching
observers in ΔΩ during an arrival bin of width Δt, per ln ν, divided by Δt T and scaled to 4π, is

    νL_ν(bin) = (mₑc² / Δln ν) (4π/ΔΩ) ((t_b − t_a)/Δt) Σ_bin w E,                         (7)

and the band luminosity of `light_curve` drops the 1/Δln ν and sums the frequency bins. T cancels.
Summing (7) over all arrival bins with the weights Δt/(t_b − t_a) gives the stationary estimator
(mₑc²/Δln ν)(4π/ΔΩ)Σ w E, which is the window mean; for a source that does not change and on a
stationary spacetime every complete bin has the expectation of the stationary spectrum.

**Error.** Given the zone assignment, the escape histories of different primaries are independent,
so Var Σ_bin = Σ_i Var X_i with X_i the total of history i in the bin. The code accumulates
Σ (wE)² over photons. Without scattering each history adds at most one photon, and the estimate
Σ x_i² has expectation Σ (Var X_i + (E X_i)²): it is conservative by Σ (E X_i)², a relative
excess of order the largest probability that one history lands in the bin (≪ 1 for a time- and
frequency-resolved bin). With scattering, children of one history in the same bin add cross terms
2Σ x_a x_b ≥ 0 that the photon-level sum misses; seed-to-seed scatter measures them.

**Complete bins.** A photon emitted at t_e arrives at t_e + Δ with Δ = (t_esc − t_e) − n̂·x_esc;
on flat space Δ = −n̂·x_e ∈ [−R, R]. An arrival time t receives the whole source only if
t − Δ ∈ [t_a, t_b] for every Δ the run can produce. The result's `interior` is [t_a + R, t_b − R]
with R the radius of the march (the sphere radius in flat space, exact there) or the emission
confinement radius R_em of a curved march (at least 100 M and 1.2 times the outermost zone), which
bounds the geometric delays and leaves 0.2 R_em for gravitational delays (4M ln(R_esc/r) ≲ 30 M);
photons on long orbits near a hole arrive later still and are a tail of the last bins.

## 6. The bitwise frozen limit

A run is the stationary run, but for the photons' time coordinate, when it has one node whose
profile is flat on the window (a snapshot on a stationary spacetime, a series of one repeated frame
on a stationary spacetime, a variable sphere of equal factors, a uniform sphere):

1. W = (t_b − t_a)·1/(t_b − t_a) = 1 exactly (x/x = 1 in IEEE arithmetic), dV·1 = dV, so the weight
   table, the counts, the stochastic rounding and every zone's envelope are the stationary ones;
2. the photon is generated by the same compiled function (`generate` of a zone is `@noinline`, so
   every program runs one method instance of it; its dot products are `@simd` reductions whose
   rounding follows the inlining context, and inlined into the window's source they differed in the
   last bit for two thirds of the torus photons) from the same stream draws, and placed at x⁰ = t_e;
3. on a stationary spacetime with time-independent matter nothing in the march reads x⁰ (the metric,
   the step, the provider), and the march runs in the same `track!` instance, because the arrival
   time is recorded by the program after `track!` returns, with the stationary sink;
4. the stationary bins are filled inside `track!` exactly as before.

So the window-mean spectra, the ledger, the counts and the children are the stationary module's bit
for bit; the time-resolved bins are new. *Correction to DESIGN T1:* with the rate convention the
equality is with the stationary spectrum itself (bit for bit); the fluence is (t_b − t_a)T times it,
one rounding. With distinct but equal frames (copies) the nodes are not merged, the zone assignment
differs, and the equality is statistical.

## 7. The emission frame

A photon is drawn in the fluid frame, ν and the pitch angle θ about the field, the azimuth φ
about the field uniform, and mapped to coordinates by the tetrad (e_(0) = u, e_(1) = b̂, e_(2),
e_(3)). Any orthonormal completion (e_(2), e_(3)) gives the same distribution, since φ is uniform;
the frame only has to be orthonormal. Gram–Schmidt on the coordinate axes x², x³ fails when an
axis is (nearly) in the span of the earlier vectors: with the field along x³, e_(1) ≈ x̂³ and the
x³ residual is a cancellation remnant. On the torus deck with a vertical field the tetrad was NaN
for 96 % of the zones (fluid at rest in its own frame: the residual vanishes exactly) and off by
O(1) for a moving fluid on Kerr (|E g Eᵀ − η| = 2.0), so those zones' photons were dropped (a
NaN energy fails E > 0 in the bins) or not null. It showed in T4: the sky pattern of a plasma
symmetric under R_z(π) was lopsided at 4–5σ on the static and the orbiting binary, the same in
six seeds, while metric, plasma, zones and every march of a photon and its rotated twin were
symmetric to rounding. The fix takes, for each of e_(2), e_(3), the axis x² (x³) unless its
residual keeps less than 10⁻² of the axis's squared norm, then the axis with the largest relative
residual (at least 1/3 for an orthonormal fluid frame, since the squared projections of the three
axes on the missing direction sum to one); the frame is orthonormal to 7e-16 on every zone.

## 8. Closed forms of the exact tests

**T2: the variable sphere.** A uniform optically thin ball of radius R, emission ∝ f(t), observer
along ẑ (any direction, by symmetry; all-sky binning uses every photon's own n̂). By (5) a photon
from z arrives at t_e − z, so the delay τ = t_arr − t_e = −z has the slab weight

    P(τ) = π(R² − τ²)/((4/3)πR³) = 3(R² − τ²)/(4R³),   |τ| ≤ R,                            (8)

and L(t) ∝ (f ∗ P)(t). For f = 1 + A sin ωt, with x = ωR,

    P̂(ω) = ∫ P(τ) e^{iωτ} dτ = (3/(4R³)) ∫_{−R}^{R} (R² − τ²) cos ωτ dτ = 3(sin x − x cos x)/x³ = 3 j₁(x)/x,

real and even, so L(t) = L₀[1 + A P̂ sin ωt]: no phase shift, the sign flipping at the zeros of j₁
(tan x = x: x = 4.4934, 7.7253, 10.9041). A `VariableSphere` with n nodes per period, factors
f_k = 1 + A sin(2πk/n), emits the piecewise-linear interpolant of those values. Its fundamental is

    c₁ = (1/P)∫₀^P f e^{−iωt}dt = (Δ/P) sinc²(ωΔ/2) Σ_k f_k e^{−2πik/n} = A sinc²(π/n)/(2i),   Δ = P/n,

using ∫ hat(t − τ_k)e^{−iωt}dt = Δ e^{−iωτ_k} sinc²(ωΔ/2) and (1/n)Σ_k sin(2πk/n)e^{−2πik/n} = 1/(2i)
(n ≥ 3). So the light curve's fundamental is A sinc²(π/n) P̂(ωR) sin ωt, exactly; the harmonics
ω(jn ± 1) are orthogonal to it over whole periods. *Correction to DESIGN T2:* the interpolation is not
a negligible error to be made small by dense sampling; for this source it is exact and known.

**T3: the light echo.** A flash at the centre at t = 0, an optically thin shell, cold electrons and
hν ≪ mₑc² (Thomson). A photon leaving the centre along ŝ scatters at r ŝ at time r with probability
dτ_r(r) (radial depth) and leaves along n̂ at angle ψ, cos ψ = ŝ·n̂, with the Thomson phase function
(3/16π)(1 + cos²ψ) (the scattering angle is ψ). By (5) its arrival time is r − n̂·(rŝ) = r(1 − μ),
μ = cos ψ. The isotropic-equivalent scattered fraction seen along n̂ per unit delay is

    Ψ_r(τ) = 4π ∫ (dΩ_s/4π) (3/16π)(1 + μ²) δ(τ − r(1 − μ)) = (3/(8r)) [1 + (1 − τ/r)²],  0 ≤ τ ≤ 2r,

normalized to 1 (∫₋₁¹(1 + μ²)dμ = 8/3), times the shell's depth. For a shell of radial depth profile
dτ_r = τ p(r) dr, to first order in τ,

    Ψ(τ) = τ ∫ p(r) Ψ_r(τ) dr.                                                             (9)

The angular distribution does not depend on the direction of the source photon, so an anisotropic
source (synchrotron, beamed about its field) gives the same all-sky echo. *Correction to DESIGN
T3:* a sharp-edged shell is resolved by the march's trapezoid depth only to O(h/Δr) at its two
edges: the scattering events of a step that crosses an edge are placed along the whole step, and
those that fall outside the plasma spawn nothing (the stationary module's rule). With 8 steps across
a uniform shell this removed 6.4 % of the scatterings (374,515 children against 400,000 expected).
The test shell is `SphericalShell`, n_e sin²(π(r − r_in)/Δ), whose density and slope vanish at both
edges, p(r) = (2/Δ) sin²(π(r − r_in)/Δ); the loss is then 0.12 % of the scattered weight at 8 steps
and 0.005 % at 32. Second-order terms (a scattered photon crossing the shell again, double
scattering) are O(τ). Two further terms enter at the converged photon count. (i) A flash of duration
δt (emission uniform over [0, δt]) shifts every arrival by its emission time, so the closed form is
(9) averaged over that shift; with δt = 10⁻³ r the first delay bin of width 0.02 r loses about
δt/(2 × 0.01 r) = 5 % of its light to the next. (ii) The march places a step's scattering events by the
step's trapezoid depth, a staircase of the profile on the step lattice; where p is small (the outer
edge, r_out − r ≲ h) the staircase overweights the last step severalfold (×3.7 in the last delay bin,
τ ∈ [2.07, 2.09] r, at h = Δ/8). The staircase converges as h → 0, and at h = Δ/32 with the flash
duration in the closed form the echo of 2 × 10⁶ photons agrees bin by bin (χ²/n = 0.996, 105 bins).

**T4: helical symmetry.** If metric and plasma are invariant under H_τ: t → t + τ, φ → φ + ωτ, then
so is the emission (2.4) and the transport, and a photon escaping along n̂ at t_arr is mapped to one
escaping along R(ωτ)n̂ at t_arr + τ (n̂·x is a rotation invariant). Hence the observed luminosity in
direction (θ, φ) satisfies

    L(t + τ, θ, φ + ωτ) = L(t, θ, φ)  for every τ,                                         (10)

and three consequences follow: the azimuth-integrated light curve of a band is constant; at fixed
(θ, φ) the light curve is periodic with P = 2π/ω and depends on t − φ/ω only; and with sectors of
width Δφ and arrival bins of width Δt = Δφ/ω, the sector j + 1 at bin k + 1 has the expectation of
sector j at bin k. *Correction to DESIGN T4:* a period P/2 at fixed (θ, φ) needs the whole system
to be invariant under the rotation by π. The equal-mass, non-spinning binary metric is, but a GRMHD
snapshot is not (its minidisks and streams differ between the holes), so the co-rotating light curve
of the snapshot has period P; P/2 holds for a plasma that is itself symmetric under R_z(π), e.g. an
axisymmetric one, for which the variability is that of the metric alone.

**T5: the stationary torus.** On Kerr a snapshot over a window, or a series of one repeated frame,
is the stationary run bit for bit (section 6), and its complete light-curve bins have the stationary
expectation. Copies of the frame (distinct objects) are separate nodes with shares 1/8, 1/4, …, 1/8
for five equally spaced frames; their run agrees with the stationary one within errors.

**T6: convergence.** Each bin is a sum over independent histories, so its relative error falls as
N^{−1/2}; the photon-level Σ(wE)² estimates the variance of a thin run to a relative O(p) bias
(section 5), so the per-bin standard errors match the scatter of independent seeds. The bias is
the stratification of the primaries: the counts of the nodes and zones are fixed (rounded, not
Poisson), and a history that lands in a bin with probability q has Var X = q(1 − q)w² against the
estimate's q w². For the variable sphere's frequency-integrated light curve a node's photons spread
over about one period of arrival bins, q ≈ 1/16, and the seed variance is 0.90–0.98 of the reported
one (eight seeds, N = 10⁴ … 10⁶).

## 9. Measured against the closed forms

Runs of `figures/paper/fig_mc_dyn_tests.jl` and `fig_mc_dyn_binary.jl` (outputs in
/data2/haiyangw/scratch/dynmc/out/), figures paper/figures/fig_mc_dyn_{tests,symmetry,binary}.pdf.

| test | measured | closed form or expectation |
|---|---|---|
| T1 sphere over [0, 30] cm, 2e6 | window mean = stationary, bitwise | bitwise |
| T5 torus 00999, snapshot and repeated frame, 2e6 | window mean, ledger, counts bitwise | bitwise |
| T5 five copies of the frame | escaped power 0.99900 ± 0.00285 of the stationary | 1 |
| T2 sphere, 32 nodes/period, 2e6 per x | amplitude/A within 1.8σ at 15 x; e.g. x = 3: 0.34727 ± 0.00448 (0.34457); x = 4.4934: 0.00099 ± 0.00449 (0); x = 6: −0.08330 ± 0.00449 (−0.08363); x = 12: −0.01640 ± 0.00551 (−0.01845) | sinc²(π/32) 3 j₁(x)/x |
| T2 phase | cosine terms within 1.1σ of 0 | 0 |
| T3 echo, 2e6, step Δ/32 | χ²/n = 0.996 over 105 bins; unspawned 5.3e-5 of the loss | τ ∫p(r)Ψ_r averaged over the flash |
| T6 N = 1e4 … 1e6, 8 seeds | σ√N constant to 3 %; seed/reported variance 0.97, 0.98, 0.95, 0.96, 0.91 | 1 − q, q ≈ 1/16 |
| T4 lopsided / half-turn symmetric deck, 1e6 | helical χ²/n 0.99 / 0.91; half period 9.56 / 0.95 | 1; ≫ 1 / 1 |
| T4 circumbinary 00455, 5e6, three bands | helical χ²/n 0.96, 1.09, 0.93 (n = 400 each); unshifted 0.94, 1.08, 2.72; residual variance 0.99 over 1200 pairs | 1 |
