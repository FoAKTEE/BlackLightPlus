# Mock runs of smressle/blacklight_binaries against blacklight

Builds: `ref-code/smressle-blacklight_binaries-038d46318a81` (the fork) and
`ref-code/c-white-blacklight` @ a7c7215 (the original), each copied to a scratch directory
and built with the shipped Makefile (`make -j16`, g++ 11.4, `-O3 -flto -fopenmp`). The fork
compiles with 56 warnings (`fork_build_warnings.txt`), the original with 1.

Data: the fork's own `scripts/generate_mock_simulation.py data/mock.athdf` (spherical
Kerr-Schild grid, 77×64×128, r from 1.915 to 51.6, `Time = 0`). Orbit files from
`write_orbit.py` in the layout read by `read_orbit_file_and_set_quantities`: a text header
`nt q` and then, per time, `t` and 18 little-endian doubles (x1, x2, a1, chi2, v1, v2);
three uniform times with a constant state. Decks: the fork's `example_simulation.input`
with `ray_integrator`, `simulation_a`, `orbit_file` and `image_time = true` changed as
listed; 64² pixels, camera at r = 50, θ = 45°, width 24 M, 8 threads.

| deck | code | holes (orbit file) | integrator | wall s (geodesics s) | NaN pixels of 4096 | notes |
|---|---|---|---|---|---|---|
| `orig_kerr0.input` | original | Kerr a = 0 | rk4 | 1.14 (0.51) | 0 | flux 2.1678e-1 |
| `orig_kerr0.input`, dp | original | Kerr a = 0 | dp | 0.82 (0.22) | 0 | flux 2.1679e-1 |
| `orig_kerr0.input`, a = 0.9, dp | original | Kerr a = 0.9 | dp | 1.17 (0.37) | 554 | the shadow lies inside the mock's r = 1.915 inner edge (`fallback_nan`) |
| `fork_single_static.input` | fork | BH1 at the origin, v = 1e-6, a = 0; BH2 massless | dp | 2.56 (1.81) | 0 | same NaN mask as the original; flux ratio 1.0000; max pixel difference 3.2e-3 of the peak; time image within 11 s |
| `fork_single_kerr09.input` | fork | as above with a1 = 0.9 ẑ | dp | 8.46 (2.30) | 554 | same NaN mask as the original at a = 0.9; flux ratio 1.0000; max pixel difference 2.6e-3 |
| `fork_single_boosted.input` | fork | as static with v1 = 0.2 ŷ | dp | 6.37 (2.23) | 427 | NaN pixels in columns 30–53, rows 20–42 (one side of the hole); flux 11% below the static case; 854,000 NaN-emissivity prints |
| `fork_binary_v02.input` | fork | BH1 (5,0,0) v = +0.2 ŷ a = 0.5 ẑ; BH2 (−5,0,0) v = −0.2 ŷ χ = 0.3 ẑ; q = 1 | rk4 | 6.91 (6.31) | 2002 | 5,435 renormalization warnings on 2,308 pixels; 21,833 NaN-emissivity prints; `debug_output_1233.txt` written (2,475 lines) |
| `fork_binary_v02.input`, dp | fork | same | dp | 92.4 (3.42) | 2397 | 88.7 s of the wall time is 4.6 million NaN-emissivity prints |
| `fork_binary_v02.input`, v = 0 orbit | fork | same holes at rest | rk4 | 74.5 (6.52) | 4096 | every pixel NaN: the radiation integrator divides by \|v\| = 0 |
| `input/example_formula.input` | fork | formula mode | — | — | — | segmentation fault (exit 139) before any output |

Logs in `logs/` (the two large stderr logs are trimmed to their first lines and counts).
The single-hole decks place a massless second hole at x = 10⁴ with v = 10⁻³ ŷ because the
radiation integrator normalizes the velocity direction without a guard. All fork runs use
`simulation_coord = sks`, under which the fluid four-velocity is built with the single
Kerr metric of `simulation_a` while the geodesics use the binary metric; the static and
Kerr decks are consistent by construction (one hole at the origin), the binary decks are
not, and their NaN counts are upper bounds on the geodesic failures (the renormalization
warnings and the trajectory file are geodesic-integrator evidence independent of the fluid).
