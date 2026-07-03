# Numerical Methods

Phase 1 contains only repository scaffolding.

## Phase 2

The Kerr metric, horizon radius, equatorial ISCO radius, circular-orbit angular
velocity, specific energy, specific angular momentum, and zero-torque efficiency
proxy are evaluated directly in double-precision Python/NumPy arithmetic.

Metric derivatives are available in two forms. `metric_derivatives` computes
centered finite differences of the covariant metric and is retained as an
independent oracle. `covariant_metric_derivatives` and
`contravariant_metric_derivatives` evaluate closed-form analytic derivatives of
the registered metric components. The analytic and finite-difference forms are
cross-checked, and the two analytic tensors are verified against the exact
identity `d(g^-1) = -g^-1 (dg) g^-1` across the valid domain. The geodesic force
term uses the analytic contravariant derivatives.

## Phase 3

The Page-Thorne zero-torque disk flux is evaluated from the conservation-law
integral rather than from the closed-form Kerr expression. Circular-orbit
quantities are evaluated on a strictly increasing radial grid. Derivatives of
`L(r)` and `Omega(r)` are computed with explicit finite-difference stencils on
the supplied nonuniform grid, and the radial integral is accumulated with the
trapezoid rule.

The implementation uses the Page-Thorne disk determinant convention
`sqrt_minus_g_disk = r` for Kerr in `G = M = c = 1`, as recorded in
`docs/equation_registry.md`. This is deliberately separate from the full
Boyer-Lindquist four-metric determinant.

## Phase 4

The local Planck function is evaluated in SI units as spectral radiance per
frequency. The implementation uses `numpy.expm1` for numerical stability and
sets the Wien-tail contribution to zero when `h nu / (k_B T) > 700`, avoiding
floating-point overflow without changing the production-relevant integral.

Bolometric checks integrate over frequency with the trapezoid rule on a
dimensionless logarithmic grid in `x = h nu / (k_B T)`. The baseline angular
emission interface returns a unit factor for `0 <= mu <= 1`, representing
isotropic comoving-frame intensity with no limb darkening.

The luminosity-dependent color-correction law exposes whether clipping occurred
through `FcolEvaluation.clipped`; production runs must record this when it
occurs.

## Phase 5

The reference ray tracer evolves Kerr null rays with Hamiltonian equations in
Boyer-Lindquist coordinates. The state vector is
`(t, r, theta, phi, p_t, p_r, p_theta, p_phi)`. Derivatives of the
contravariant metric in the force term are analytic (Phase 2). Two integrators
are provided:

- `trace_ray` is the explicit fixed-step RK4 reference with event checks after
  each accepted step. It is retained for regression and as a baseline.
- `trace_ray_adaptive` is an embedded Dormand-Prince RK45 integrator with a
  PI-style step-size controller targeting a relative/absolute error tolerance
  (`rtol`/`atol`). The disk-crossing event is refined by bisection on the
  sub-step rather than by linear interpolation, so the recorded emission radius
  and near-zero event residual are controlled to `event_tol`. `rtol` provides a
  single-parameter convergence knob for tolerance studies.

At `rtol = 1e-9` the adaptive tracer reproduces the fixed-step disk-hit radius
to better than `1e-4 r_g` while using far fewer right-hand-side evaluations, and
it conserves energy, axial angular momentum, and the Carter constant to the
declared invariant tolerances. Rays approaching the polar axis are rejected
explicitly (`geodesics._AXIS_BUFFER`) because the Boyer-Lindquist contravariant
metric diverges there.

The current event layer records exactly one terminal outcome:

- `DISK_HIT`
- `HORIZON_CAPTURE`
- `ESCAPED`
- `MAX_STEPS`
- `NUMERICAL_FAILURE`

The default horizon buffer is intentionally conservative (`r_stop = 1.1 r_+`)
for the scalar Boyer-Lindquist reference integrator. Smaller buffers require a
separate convergence study because near-horizon coordinate stiffness can
dominate the null-constraint error.

The observer tetrad is static and intended for large observer radius outside
the ergosphere. The screen-coordinate convention is documented in
`docs/equation_registry.md` and is not yet the final transfer-map camera model.

## Phase 6

Transfer maps are built by tracing each screen coordinate independently with the
Phase 5 scalar ray tracer. Only `DISK_HIT` rays become transfer records; all
outcomes are counted. Each record stores screen coordinates, solid angle,
emission radius, azimuth, redshift factor, physical emission angle,
Hamiltonian diagnostic, and step count.

The redshift factor is computed from the invariant measured frequency
`nu = -p_mu u^mu`. The emitter is assumed to be on an equatorial circular
geodesic. Observed spectra use `I_nu / nu^3` invariance:
`I_obs(nu_obs) = g^3 I_em(nu_obs / g)`, summed over screen solid angle.
The emission angle is computed as
`emission_mu = |p_mu n^mu| / (-p_mu u_em^mu)`.

Transfer-map cache files are compressed NumPy archives with JSON metadata. They
are intended as deterministic intermediate products, not publication artifacts.

## Phase 8

The Agol-Krolik inner-edge stress extension is evaluated as an additive
one-face surface-flux term on the same radial grid as the Page-Thorne baseline.
The baseline profile is computed first, then the stress contribution is added
without changing the ISCO radius or the circular-orbit functions.

The added luminosity check uses the same infinity convention as the Phase 3
Page-Thorne validation: the two-face ring contribution is multiplied by the
circular-orbit specific energy before radial integration,
`4 pi r F_extra E(r) dr`. With this convention, the Agol-Krolik amplitude
integrates to `mdot Delta_eta`.

The implementation enforces `0 <= Delta_eta <= 0.1` as the initial controlled
weak-stress domain. It performs no clipping: invalid values raise explicit
errors.

## Phase 9

Synthetic spectra are represented on detector-independent logarithmic energy
bins in keV. Photon-flux density callables are integrated with a midpoint rule
on each bin for this development layer. The stored bin energy flux is
`E_center * photon_flux`, in `keV cm^-2 s^-1 bin^-1`.

The Gaussian debug-noise model draws from independent normal distributions with
standard deviation equal to a fixed fraction of the noiseless bin energy flux.
The Poisson model draws independent counts from bin photon flux multiplied by a
scalar effective area and exposure. These two noise models are kept separate,
and their likelihood functions use the matching statistical model.

Seeds are derived hierarchically with a stable BLAKE2 hash:
`master_seed -> condition_seed -> epoch_seed -> noise_seed`. The generated
`SeedManifest` is stored on every synthetic epoch so datasets can be replayed.

The bounded scalar likelihood maximizer is a deterministic golden-section
search used only for Phase 9 same-model recovery tests. Phase 10 adds the
first shared posterior API, optimizer, sampler adapter, and calibration
diagnostics.

## Phase 10

The inference layer exposes bounded uniform priors, a common posterior API, a
deterministic coordinate-pattern optimizer, and an internal random-walk
Metropolis adapter. The optimizer evaluates coordinate-wise proposals at a
shrinking step size until the maximum step is below tolerance or the iteration
budget is exhausted. Proposals outside the prior box are rejected by the prior
returning `-inf`.

The Metropolis adapter runs repeated chains from explicit initial positions
with Gaussian random-walk proposals. The implementation records samples,
log-posterior values, and per-chain acceptance fractions. It is a validation
baseline for Phase 10, not yet the final production sampler.

Diagnostics include basic split-chain R-hat and an initial-positive
autocorrelation estimate of effective sample size. Simulation-based calibration
is represented by one-dimensional posterior ranks, rank histograms, and
equal-tailed 68% and 95% interval coverage.

The frozen Phase 10 prior table is `paper/tables/priors.csv`.

## Phase 11

The coarse screening campaign uses a lightweight detector-independent proxy
spectrum to exercise the pipeline before expensive confirmatory simulations.
For each condition, Gaussian relative-error synthetic spectra are generated for
20 deterministic noise replicates. A one-dimensional spin posterior is
evaluated on a fixed grid over the frozen spin prior, normalized with a stable
log-sum-exp style shift, and summarized with posterior mean, MAP, equal-tailed
68%/95% intervals, and a chi-square-per-degree-of-freedom fit statistic.

Condition summaries retain both completed and failed replicates. A condition is
selected for confirmatory refinement when any frozen criterion is met:
high bias, low identifiability, poor fit, or nonzero failure rate. The generated
confirmatory protocol records these thresholds and explicitly prohibits
post-hoc condition removal without a new protocol version.

## Phase 12

The confirmatory runner consumes the locked Phase 11 condition list and frozen
protocol. It generates deterministic blinded IDs, writes hidden truths to a
separate file, freezes the analysis settings, fits the base replicate set, then
joins hidden truths only for the final unblinded summary table.

For each condition, Phase 12 uses 100 deterministic noise replicates and a
higher-resolution spin-grid rerun. The corrected
`phase12_ray_traced_transfer_v4` run uses 81 base spin grid points and 121
higher-resolution spin grid points. A condition is marked numerically stable
when the absolute difference between base and higher-resolution mean bias is no
larger than the frozen tolerance.

The corrected backend uses the registered Page-Thorne and Agol-Krolik disk flux
profiles, a logarithmic radial grid from ISCO to the configured outer disk
radius, and a diluted-blackbody local thermal spectrum. It traces a 5x5
cell-centered image-plane screen for each spin/inclination transfer map,
records disk-hit redshift factors, solid angles, and physical emission angles,
then sums the observed spectrum with `I_nu / nu^3` invariance and
electron-scattering limb darkening. Transfer maps are cached within each
resolution run by spin, inclination, and screen size, so deterministic noise
replicates reuse the same ray-traced transfer records.

The v4 run replaces the Phase 11 proxy and the earlier inclination-projected
Phase 12 backend. It still uses the scalar reference ray tracer. Returning
radiation and photon-capture outcomes are diagnosed in Phase 12.5 but are not
iteratively reprocessed into the confirmatory likelihood.

### Corrected image plane (`phase12_ray_traced_transfer_v5`)

The v4 configuration traced a 5x5 (25-ray) screen over a narrow, hand-tuned,
off-center window (`alpha in [-8, 8]`, `beta in [35, 65]`) with the observer at
`r = 100`. This window sampled only emission radii near `37-53 r_g` and missed
the hot inner disk that carries most of the spin information, and at `r = 100`
the flat-screen small-angle mapping is inaccurate for wide-angle rays. The v5
configuration corrects both defects:

- The observer is placed at `r = 1000` so the flat-screen small-angle mapping
  stays accurate over the whole image.
- The screen spans `+/- 1.4 * disk_outer_radius` symmetrically
  (`full_disk_screen_axes`), capturing the full disk image from the ISCO to the
  outer edge with no boundary clipping. For `a = 0.5`, `i = 40 deg` the corrected
  screen samples emission radii from `~4.3 r_g` (the ISCO) to `~80 r_g`.
- The image is traced with the adaptive Dormand-Prince integrator at a
  convergence-justified `64x64` resolution. A resolution study on the corrected
  full-disk screen gives relative-L1 spectrum differences versus a `112x112`
  reference of `1.9%` (24x24), `0.73%` (40x40), `0.38%` (64x64), and `0.12%`
  (88x88).

The v5 spectrum is also absolutely normalized to a physical system rather than
scaled by an arbitrary temperature parameter. The mass (`mass_msun`) sets the
gravitational radius and, with the Eddington ratio and radiative efficiency, the
accretion rate; the Page-Thorne SI flux sets the effective-temperature profile
in kelvin; and the distance (`distance_kpc`) sets the observed flux level. The
transfer map is built with `observer_distance` equal to the astronomical
distance in gravitational radii so its solid angle is the physical observer
solid angle, giving the correct `r_g^2` and `D^-2` scaling. For the fiducial
ten-solar-mass, 8 kpc system the peak effective temperature is `~0.5 keV`, in
the soft X-ray band. Because the fit uses noise anchored to the true spectrum,
the absolute normalization (which depends on mass and spin through the ISCO) now
carries spin information in addition to the spectral shape.

The v4 confirmatory artifacts under `data/processed/confirmatory` and the paper
figures/tables derived from them predate this correction and must be regenerated
with the v5 configuration before they are used for any scientific claim.

## Phase 12.5

The transfer-validation command runs three checks. The screen-convergence
campaign compares 3x3, 5x5, and 7x7 cell-centered image-plane maps against the
7x7 reference for two benchmark spin/inclination pairs. The
capture/returning-radiation diagnostic samples locally emitted photons from
selected annuli in the circular-emitter frame and classifies terminal outcomes.
The external comparison harness compares production transfer records against a
user-supplied external ray-tracer CSV when one is available.
`scripts/audit_external_raytracers.py` records whether candidate external
backends are locally available. This audit does not establish agreement by
itself; it only documents whether an independent backend is present for a
future comparison run.

## Phase 13

Paper figures are generated by `scripts/make_figures.py` from final processed
CSV outputs and deterministic display calculations. Figure source data that are
not already present as processed campaign outputs are written to
`paper/tables/figure*_source.csv` files before plotting. Paper summary tables
and `paper/claim_audit.md` are generated by `scripts/make_tables.py`.

The claim audit is intentionally conservative: claims with no supporting final
output are retained as `UNSUPPORTED` rows rather than removed.

## Phase 13.5

The pre-manuscript multi-epoch comparison pairs locked Phase 12 conditions that
share spin, inclination, true/fitted color-correction assumptions, and
inner-stress setting, while using the two selected luminosities as epochs. For
each group, `scripts/run_multi_epoch.py` regenerates deterministic Gaussian
noise realizations from the same `ray_traced_transfer` backend, fits each epoch
separately on the frozen spin grid, and fits the paired epochs jointly with one
shared spin parameter.

The comparison is a fixed-luminosity shared-spin benchmark. It does not fit
epoch-level luminosity as a nuisance parameter and does not add detector
response, background, or calibration uncertainty. Its outputs support a
multi-epoch versus single-epoch comparison, not a blanket claim that joint
fitting improves spin recovery.

`scripts/reproduce_all.py --mode archive` regenerates paper tables and figures
from archived processed outputs, then writes a release run manifest and SHA-256
checksum CSV under `data/processed/release`. `--mode full` is wired to rerun the
heavy validation, confirmatory, transfer-validation, external-backend audit,
and multi-epoch commands before regenerating paper artifacts.

## Joint Multi-Parameter Inference

The earlier campaign fitted only spin on a one-dimensional grid while holding
inclination, accretion rate, and color correction fixed at the model values.
That scan cannot represent parameter degeneracies or a marginalized
uncertainty. `src/kerrdisk/joint_inference.py` adds a joint fit over several
parameters (for example spin, Eddington ratio, color correction, and
inclination) using the affine-invariant ensemble sampler `emcee`. The spin
posterior is obtained by marginalizing the nuisance parameters. `make_spectrum_posterior`
builds a Gaussian-spectrum posterior over a forward model with fixed data
variance; `run_emcee_sampler` runs a seeded, reproducible ensemble and returns
chains in the shared `ChainResult` format so the existing R-hat and
effective-sample-size diagnostics apply.

Freeing the color correction inflates the marginal spin interval relative to a
fixed-color fit, which is the spin/color-correction degeneracy the study is
designed to expose.

### Fast forward model (spectral grid emulator)

A full ensemble fit needs thousands of forward evaluations, which is infeasible
if each evaluation ray-traces a new transfer map. The transfer map depends only
on spin and inclination, while the Eddington ratio and color correction are
cheap transforms of the effective-temperature profile and local emission at a
fixed geometry. `src/kerrdisk/emulator.py` exploits this: `build_spectral_grid`
evaluates a spectral backend once on a regular parameter grid (tracing the
costly transfer maps only at the nodes), and `SpectralGrid` interpolates
multilinearly at every posterior evaluation. This is the standard
precomputed-table strategy used by continuum-fitting spin codes.

### Simulation-based calibration on the actual model

`run_spectrum_sbc` performs simulation-based calibration on the actual spectral
forward model rather than a one-dimensional analytic surrogate. For each
simulation it draws a truth from the prior, generates a noisy spectrum, samples
the joint posterior with ensemble walkers dispersed across the prior, and
records the rank of the true value within its marginal posterior. The aggregate
rank histogram and 68 / 95 percent interval coverage reuse the Phase 10 SBC
summary. Rigorous calibration with many simulations and a production-resolution
grid is an offline job; the packaged tests are fast smoke checks with loose
calibration bands.

### Migration status

The joint fit, spectral-grid emulator, and model-based SBC are validated
capabilities. Migrating the confirmatory and multi-epoch campaigns from the
one-dimensional spin scan to the joint fit over a precomputed emulator (built
from the corrected v5 ray-traced backend) is the next production step and, like
the v5 campaign regeneration, is a compute job rather than a code change.
