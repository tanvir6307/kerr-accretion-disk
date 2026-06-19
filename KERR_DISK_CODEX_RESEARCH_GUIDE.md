# Kerr Accretion-Disk Research: Codex-Ready Guide from Zero to Publication

**Working project name:** `KerrDisk-UQ`  
**Primary scientific paper:** *Can Multi-Epoch Thermal Spectra Reduce Systematic Bias in Kerr Black-Hole Spin Inference?*  
**Secondary software-paper option:** *KerrDisk-UQ: An Open Python Framework for Relativistic Thin-Disk Spectra and Uncertainty Quantification*  
**Recommended language:** Python 3.12+  
**Research level:** Computational astrophysics; relativistic thin accretion disks; synthetic observations; Bayesian/likelihood-based inference  
**Status:** Project specification. Do not treat any intended result as established until the complete validation and analysis pipeline has passed.

---

## 0. Read This First

This file is an execution contract for Codex. Work phase by phase. Do not skip validation because a plot “looks correct.”

### Non-negotiable rules for Codex

1. **Never invent an equation, citation, numerical benchmark, instrument response, or scientific result.**
2. Every physical equation implemented in production code must be connected to a cited source in `docs/equation_registry.md`.
3. Every important numerical result must be reproducible from a committed configuration file and a single command.
4. Do not describe a result as “novel” until the literature-gap audit in Phase 0 is completed.
5. Do not use generated numbers in the manuscript unless they come from versioned output files.
6. Never silently clip, replace, smooth, or discard failed simulations. Record failures with reasons.
7. Use double precision by default. Any acceleration must be shown not to alter scientific conclusions.
8. The scientific manuscript must distinguish:
   - established theory,
   - numerical implementation,
   - validation evidence,
   - new simulation results,
   - interpretation,
   - limitations.
9. AI-generated code is not trusted by default. It must pass tests, convergence studies, and an independent comparison.
10. Maintain an `AI_USAGE.md` file describing how Codex was used and how generated work was checked.

### Stop conditions

Stop and report rather than continuing when:

- an equation cannot be verified from a primary or authoritative source;
- two validation routes disagree beyond their declared tolerance;
- the same scientific question has already been answered with essentially the same model and parameter study;
- the thin-disk assumptions are violated by the selected parameter range;
- posterior recovery or coverage tests fail;
- output changes materially when numerical resolution is increased.

---

# Part I — Scientific Decision

## 1. Recommended Research Question

### Main question

> Under what combinations of black-hole spin, inclination, luminosity, spectral-hardening uncertainty, and weak inner-edge stress does a standard zero-torque Novikov–Thorne fit return a biased Kerr spin, and can joint multi-epoch fitting reduce that bias?

This is stronger than a generic project showing that spin changes disk temperature. That dependence is already well known.

### Core simulation design

Generate fully relativistic thermal spectra from a Kerr thin disk over multiple accretion states. Treat the black-hole spin, mass, distance, and inclination as shared quantities across epochs. Allow the accretion rate and possibly the color-correction factor to vary between epochs. Generate synthetic data with controlled model departures, then fit them with simpler models and measure:

- spin bias;
- uncertainty calibration;
- parameter degeneracies;
- posterior coverage;
- conditions where multi-epoch fitting helps;
- conditions where it does not help.

### Primary hypotheses

Use these as pre-registered hypotheses, not conclusions.

- **H1:** Fixing an incorrect color-correction factor produces a spin bias that increases toward high spin and high inclination.
- **H2:** Weak nonzero inner-edge stress can mimic a smaller ISCO and therefore bias spin upward when fitted with a zero-torque model.
- **H3:** Joint fitting of several luminosity states reduces random uncertainty but does not automatically remove systematic bias.
- **H4:** A luminosity-dependent color-correction model improves spin recovery compared with a fixed color-correction model when the true spectra contain such evolution.
- **H5:** There is a measurable domain in parameter space where spin is not identifiable from the thermal continuum alone, even at high signal-to-noise ratio.

### Minimum publishable scientific contribution

The paper must deliver all of the following:

1. A verified Kerr thin-disk spectral simulator.
2. A documented, physically controlled model-misspecification experiment.
3. A broad and predeclared simulation matrix, including prograde and retrograde spin where physically relevant.
4. Quantitative bias and coverage maps, not only example spectra.
5. Multi-epoch versus single-epoch comparisons.
6. Numerical convergence and independent validation.
7. Open code, configurations, derived data, and figure-generation scripts.
8. A clear statement of what is new relative to earlier continuum-fitting uncertainty studies.

If the final work only reproduces standard Novikov–Thorne profiles, it is a learning project, not yet a research paper.

---

## 2. Scope and Physical Assumptions

### Included in the main project

- Kerr spacetime in Boyer–Lindquist coordinates.
- Dimensionless spin \(a_\ast = Jc/(GM^2)\).
- Equatorial, geometrically thin, optically thick disk.
- Circular geodesic orbital motion outside the inner edge.
- Steady-state Novikov–Thorne/Page–Thorne dissipation.
- Zero-torque baseline at the ISCO.
- Optional controlled inner-edge-stress extension based on an established published prescription.
- Local diluted blackbody emission with color correction.
- Relativistic energy shifts, Doppler boosting, gravitational redshift, lensing, and projected area through ray tracing.
- Synthetic multi-epoch spectra.
- Statistical fitting and uncertainty quantification.
- Optional instrument folding after the detector-independent calculation is validated.

### Excluded from the first paper

- Full GRMHD evolution.
- MRI turbulence from first principles.
- Radiation-MHD.
- Thick/slim disks at high Eddington ratio.
- Corona and Comptonization as a primary physical model.
- Reflection spectroscopy and iron-line modeling.
- Polarization transport.
- Returning radiation in the baseline.
- Disk self-gravity.
- Warped or tilted disks.
- Time-dependent fluid dynamics.

These may be future projects. Do not add them before the baseline project is complete.

### Validity domain

Use the thin-disk model only in a conservative luminosity domain. The exact cut must be justified in the manuscript. A practical initial design is:

\[
0.03 \le L/L_{\rm Edd} \le 0.30.
\]

Flag all runs outside the adopted domain. Do not hide them inside the main analysis.

---

## 3. Novelty Gate

Complete this before claiming a final paper title.

### Required literature groups

Create `literature/gap_matrix.csv` with at least 40 directly relevant papers, including:

1. Kerr circular geodesics and ISCO.
2. Novikov–Thorne/Page–Thorne disk theory.
3. Relativistic transfer functions and ray tracing.
4. Continuum-fitting spin measurements.
5. GRMHD tests of the Novikov–Thorne assumption.
6. Color-correction uncertainty.
7. Nonzero stress at the ISCO.
8. Finite disk thickness.
9. Multi-epoch or hierarchical continuum fitting.
10. Open-source Kerr spectral or ray-tracing software.
11. Recent work from 2024–2026.

### Columns required in `gap_matrix.csv`

```text
citation_key,title,year,model,ray_tracing,parameters_varied,
single_or_multi_epoch,systematics_studied,inference_method,
open_code,main_result,overlap_with_our_plan,remaining_gap,notes
```

### Novelty decision

Write `literature/novelty_decision.md` answering:

- Has joint multi-epoch spin recovery under simultaneous color-correction and inner-stress misspecification already been mapped?
- Did earlier work report full bias and frequentist/Bayesian coverage maps?
- Did earlier work include retrograde through near-extremal prograde spin?
- Did earlier work publish reusable open Python code and all configurations?
- What exact new scientific statement can this project support?

### Pivot rules

If the exact primary question is already covered, choose one of these pivots:

#### Pivot A — Certified spectral emulator

Build a surrogate for ray-traced Kerr spectra with a declared maximum interpolation error, then demonstrate unbiased posterior recovery and a measured speedup.

#### Pivot B — Multi-epoch identifiability

Focus on when shared-parameter multi-epoch fitting can or cannot break the \(a_\ast\)-\(f_{\rm col}\)-inclination degeneracy.

#### Pivot C — Retrograde-to-prograde systematic map

Study the complete spin interval and identify asymmetric inference failures that are missed by prograde-only grids.

#### Pivot D — Luminosity-dependent atmosphere mismatch

Use several physically motivated \(f_{\rm col}(L/L_{\rm Edd})\) laws and determine which spin conclusions remain stable.

Do not pivot to an arbitrary modified-gravity metric merely because it is easy to change the Kerr metric. A new metric parameter alone does not guarantee a meaningful astrophysics paper.

---

# Part II — Repository and Engineering Contract

## 4. Repository Layout

Create exactly this high-level structure:

```text
kerrdisk-uq/
├── AGENTS.md
├── AI_USAGE.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── docs.yml
│       └── release.yml
├── configs/
│   ├── validation/
│   ├── production/
│   ├── inference/
│   └── figures/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── equation_registry.md
│   ├── assumptions.md
│   ├── numerical_methods.md
│   ├── validation.md
│   ├── user_guide.md
│   └── api/
├── literature/
│   ├── gap_matrix.csv
│   ├── novelty_decision.md
│   └── references.bib
├── paper/
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   ├── tables/
│   ├── references.bib
│   ├── cover_letter.tex
│   └── response_to_reviewers.tex
├── scripts/
│   ├── run_validation.py
│   ├── run_grid.py
│   ├── run_inference.py
│   ├── make_figures.py
│   ├── make_tables.py
│   ├── archive_release.py
│   └── reproduce_all.py
├── src/
│   └── kerrdisk/
│       ├── __init__.py
│       ├── constants.py
│       ├── units.py
│       ├── config.py
│       ├── metric.py
│       ├── geodesics.py
│       ├── circular_orbits.py
│       ├── isco.py
│       ├── disk_flux.py
│       ├── atmosphere.py
│       ├── raytrace.py
│       ├── spectrum.py
│       ├── synthetic.py
│       ├── likelihood.py
│       ├── inference.py
│       ├── diagnostics.py
│       ├── io.py
│       ├── plotting.py
│       └── cli.py
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    ├── convergence/
    └── data/
```

### Architecture rules

- Physics functions must be pure where practical.
- Unit conversion must be isolated from dimensionless geometric-unit calculations.
- Plotting code must not calculate physics.
- Manuscript tables must be produced from machine-readable files.
- Every run must write:
  - resolved configuration;
  - git commit;
  - package versions;
  - random seed;
  - start/end timestamps;
  - warnings;
  - convergence status;
  - checksum manifest.
- Never overwrite a production run. Use content-addressed or timestamped output directories.

---

## 5. Environment

### Suggested dependencies

Core:

```text
numpy
scipy
astropy
h5py
xarray
pandas
pydantic
pyyaml
matplotlib
numba
rich
typer
```

Inference:

```text
emcee
dynesty
arviz
```

Engineering:

```text
pytest
pytest-cov
hypothesis
ruff
mypy
pre-commit
sphinx
```

Optional:

```text
jax
corner
sherpa
```

Do not add optional dependencies until a specific requirement exists.

### Initial commands

```bash
mkdir kerrdisk-uq
cd kerrdisk-uq
git init
uv init --package
uv add numpy scipy astropy h5py xarray pandas pydantic pyyaml matplotlib numba rich typer
uv add --dev pytest pytest-cov hypothesis ruff mypy pre-commit sphinx
uv add --optional inference emcee dynesty arviz corner
uv lock
uv run pytest
```

### Quality commands

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/kerrdisk
uv run pytest -q
uv run pytest --cov=src/kerrdisk --cov-report=term-missing
```

### Minimum quality thresholds

- Core scientific modules: at least 90% branch coverage.
- Entire package: at least 85% branch coverage.
- No untyped public production API.
- No ignored failing tests.
- No warning suppression without written justification.

---

# Part III — Physics Specification

## 6. Units and Conventions

Use geometric units internally:

\[
G=M=c=1.
\]

Define:

\[
r_g=\frac{GM}{c^2}, \qquad t_g=\frac{GM}{c^3}.
\]

The dimensionless Kerr spin is:

\[
a_\ast=\frac{Jc}{GM^2}, \qquad -1<a_\ast<1.
\]

Use `a_star` for the dimensionless spin in the public API. Do not use the same variable for both dimensional \(a=J/(Mc)\) and dimensionless \(a_\ast\).

### Sign convention

- Positive `a_star`: prograde disk relative to the black-hole spin.
- Negative `a_star`: retrograde disk.
- The disk orbital direction is fixed as positive azimuthal motion. The sign of `a_star` carries alignment.

Document this convention in every configuration schema and manuscript table.

---

## 7. Kerr Metric

In Boyer–Lindquist coordinates \((t,r,\theta,\phi)\), define:

\[
\Sigma=r^2+a^2\cos^2\theta,
\]

\[
\Delta=r^2-2r+a^2,
\]

\[
A=(r^2+a^2)^2-a^2\Delta\sin^2\theta.
\]

Implement covariant and contravariant metric tensors and their derivatives. Test:

\[
g_{\mu\alpha}g^{\alpha\nu}=\delta_\mu{}^\nu.
\]

### Horizon

\[
r_+=1+\sqrt{1-a_\ast^2}.
\]

Production ray tracing must terminate safely before crossing the horizon. Use a configurable buffer, for example:

\[
r_{\rm stop}=r_+(1+\epsilon_h),
\]

with a convergence test for \(\epsilon_h\).

---

## 8. ISCO

Implement the analytic equatorial Kerr ISCO expression using:

\[
Z_1=1+(1-a_\ast^2)^{1/3}
\left[(1+a_\ast)^{1/3}+(1-a_\ast)^{1/3}\right],
\]

\[
Z_2=\sqrt{3a_\ast^2+Z_1^2},
\]

\[
r_{\rm ISCO}=3+Z_2-\operatorname{sign}(a_\ast)
\sqrt{(3-Z_1)(3+Z_1+2Z_2)}.
\]

Handle \(a_\ast=0\) explicitly.

### Mandatory benchmark values

| \(a_\ast\) | \(r_{\rm ISCO}/r_g\) |
|---:|---:|
| -0.9 | 8.717352279606489 |
| -0.5 | 7.554584714512358 |
| 0.0 | 6.000000000000000 |
| 0.5 | 4.233002529530826 |
| 0.9 | 2.320883041761887 |
| 0.998 | 1.236970655175185 |

Use a tolerance of \(10^{-12}\) for direct double-precision evaluation away from the extremal limit.

---

## 9. Circular Equatorial Orbits

For the adopted sign convention, implement:

\[
\Omega=\frac{1}{r^{3/2}+a_\ast},
\]

\[
E=
\frac{r^{3/2}-2r^{1/2}+a_\ast}
{r^{3/4}\sqrt{r^{3/2}-3r^{1/2}+2a_\ast}},
\]

\[
L=
\frac{r^2-2a_\ast r^{1/2}+a_\ast^2}
{r^{3/4}\sqrt{r^{3/2}-3r^{1/2}+2a_\ast}}.
\]

Verify the sign convention against the primary literature before finalizing the implementation.

### Efficiency benchmark

For a zero-torque disk without photon capture:

\[
\eta_{\rm NT}=1-E(r_{\rm ISCO}).
\]

Mandatory approximate values:

| \(a_\ast\) | \(\eta_{\rm NT}\) |
|---:|---:|
| -0.9 | 0.03899834565 |
| -0.5 | 0.04514222695 |
| 0.0 | 0.05719095842 |
| 0.5 | 0.08211799334 |
| 0.9 | 0.15575299199 |
| 0.998 | 0.32099416562 |

Do not compare the \(a_\ast=0.998\) value directly with a photon-capture-corrected Thorne efficiency without explaining the difference.

---

## 10. Relativistic Thin-Disk Flux

Use the Page–Thorne conservation-law form. Implement it first as a numerical integral rather than copying a long closed-form expression.

A general form is:

\[
F(r)=
\frac{\dot M}{4\pi\sqrt{-g}}
\frac{-\Omega_{,r}}{(E-\Omega L)^2}
\int_{r_{\rm in}}^r
(E-\Omega L)L_{,r'}\,dr',
\]

subject to the exact metric-area convention used in the selected derivation.

### Critical instruction

Before coding, write `docs/equation_registry.md` containing:

- source;
- equation number;
- coordinate convention;
- definition of \(\sqrt{-g}\);
- whether \(F\) is one-face or two-face flux;
- whether radius is dimensional or in \(GM/c^2\);
- conversion back to cgs/SI;
- expected asymptotic behavior.

Do not proceed until dimensional analysis is complete.

### Required physical checks

- \(F(r_{\rm ISCO})=0\) for the zero-torque model.
- \(F(r)>0\) outside the ISCO.
- At large radius, recover the Newtonian scaling \(F\propto r^{-3}\).
- Integrated luminosity approaches \(\eta_{\rm NT}\dot M c^2\), within the exact assumptions of the implementation.
- Results are stable under radial-grid refinement.

---

## 11. Temperature and Local Emission

Effective temperature:

\[
T_{\rm eff}(r)=\left[\frac{F(r)}{\sigma_{\rm SB}}\right]^{1/4}.
\]

Use a diluted blackbody:

\[
I_{\nu,{\rm em}}=
\frac{1}{f_{\rm col}^4}
B_\nu\!\left(f_{\rm col}T_{\rm eff}\right),
\]

where \(f_{\rm col}\) is the color-correction factor.

### Baseline emission assumptions

- isotropic emission in the comoving frame;
- no limb darkening in the baseline;
- no returning radiation;
- no Compton tail;
- no absorption.

Add each extension separately and test it. Do not combine unvalidated effects.

### Color-correction models

Implement:

1. `ConstantFcol(f_col)`.
2. `EpochwiseFcol(values)`.
3. `LuminosityLawFcol(f0, f1, pivot, lower, upper)`.

A controlled law for simulation experiments may be:

\[
f_{\rm col}(\ell)=
\operatorname{clip}\left[
f_0+f_1\log_{10}\left(\frac{\ell}{\ell_0}\right),
f_{\min},f_{\max}
\right],
\]

where \(\ell=L/L_{\rm Edd}\).

State clearly that this is a controlled sensitivity model unless its parameters come from an atmosphere calculation.

---

## 12. Optional Inner-Edge Stress

Do not invent a torque term. Implement an established prescription, such as a published nonzero-stress thin-disk model, with the original notation documented.

Parameterize the extension using a physically interpretable quantity, preferably an additional radiative efficiency:

\[
\Delta\eta \ge 0.
\]

Requirements:

- zero parameter exactly recovers the Page–Thorne baseline;
- added dissipation is nonnegative over the chosen domain;
- integrated additional luminosity matches \(\Delta\eta\dot M c^2\);
- large-radius behavior agrees with the source model;
- parameter limits are conservative and justified.

The primary paper should call this a controlled nonzero-stress model, not a full magnetic-disk simulation.

---

## 13. Kerr Ray Tracing

### Production method

Backward ray trace from an observer screen at large radius to the equatorial disk.

A robust implementation may use the Hamiltonian system:

\[
H=\frac{1}{2}g^{\mu\nu}p_\mu p_\nu=0,
\]

\[
\frac{dx^\mu}{d\lambda}=g^{\mu\nu}p_\nu,
\]

\[
\frac{dp_\mu}{d\lambda}
=-\frac{1}{2}\partial_\mu g^{\alpha\beta}p_\alpha p_\beta.
\]

Alternative first-order Carter equations may be used if turning points are handled correctly.

### Ray outcomes

Every ray must end in exactly one state:

- `DISK_HIT`
- `HORIZON_CAPTURE`
- `ESCAPED`
- `MAX_STEPS`
- `NUMERICAL_FAILURE`

Record outcome statistics for every image.

### Invariants and numerical diagnostics

Track:

- null condition \(H=0\);
- energy \(E_\gamma=-p_t\);
- axial angular momentum \(L_{\gamma,z}=p_\phi\);
- Carter constant or an equivalent invariant;
- step count;
- event residual at disk crossing.

Set declared tolerances and perform resolution studies.

### Observer screen

Implement a local orthonormal tetrad at large observer radius. Map screen coordinates \((\alpha,\beta)\) to the initial photon four-momentum.

Do not use an unverified “camera formula” copied from a secondary website.

### Disk intersection

Detect a sign change in \(\theta-\pi/2\). Refine the event root. Accept a hit only when:

\[
r_{\rm in}\le r_{\rm hit}\le r_{\rm out}.
\]

### Redshift factor

Use invariance of \(I_\nu/\nu^3\):

\[
I_{\nu,{\rm obs}}=g^3 I_{\nu,{\rm em}},
\qquad
g=\frac{\nu_{\rm obs}}{\nu_{\rm em}}
=\frac{-p_\mu u^\mu_{\rm obs}}
{-p_\mu u^\mu_{\rm em}}.
\]

For the distant static observer, simplify only after deriving the expression from the four-velocities.

### Observed flux

\[
F_{\nu,{\rm obs}}
=\int I_{\nu,{\rm obs}}\,d\Omega_{\rm obs}.
\]

For a flat screen at distance \(D\):

\[
d\Omega_{\rm obs}\simeq\frac{d\alpha\,d\beta}{D^2}.
\]

### Performance strategy

1. Implement a correct scalar reference integrator.
2. Validate.
3. Vectorize screen-ray initialization.
4. Parallelize independent rays.
5. Add Numba only after regression tests exist.
6. Cache transfer maps for fixed \((a_\ast,i,r_{\rm out})\).
7. Never optimize by loosening physics tolerances without a documented error study.

---

# Part IV — Validation Program

## 14. Validation Hierarchy

Validation must proceed in this order.

### Level 1 — Algebra and units

- Metric inverse identity.
- Horizon location.
- ISCO values.
- Circular-orbit \(E,L,\Omega\).
- Efficiency.
- Unit conversions.
- Blackbody normalization.

### Level 2 — Limiting cases

- Schwarzschild limit \(a_\ast=0\).
- Large-radius Newtonian limit.
- Face-on symmetry.
- Zero inner torque.
- Low-inclination red/blue symmetry behavior.
- Distance scaling \(F_\nu\propto D^{-2}\).
- Mass/accretion-rate scaling where analytically expected.

### Level 3 — Geodesic invariants

- Null Hamiltonian remains near zero.
- Conserved quantities remain stable.
- Shadow boundary agrees with analytic Kerr critical curves or a trusted independent implementation.
- Ray outcomes converge with step tolerance.

### Level 4 — Disk flux

- Flux vanishes at the ISCO for zero torque.
- Luminosity integral agrees with the expected efficiency.
- Radial profiles match independent calculations or digitized published benchmarks.
- Grid refinement changes key quantities below tolerance.

### Level 5 — Spectrum

- Integrating the image reproduces direct axisymmetric calculations in a no-bending test.
- Face-on Schwarzschild spectra agree with an independent implementation.
- Flux scales correctly with distance.
- Spectral peak shifts in the physically expected direction under controlled changes, but qualitative direction alone is not validation.

### Level 6 — Inference

- Simulated data generated and fit by the same model recover injected parameters.
- Posterior rank/coverage tests are approximately calibrated.
- Optimization and sampling agree for unimodal test cases.
- Different seeds do not change conclusions.
- Misspecified-model bias is separated from numerical bias.

---

## 15. Validation Acceptance Criteria

Create `configs/validation/acceptance.yaml`.

Suggested initial criteria:

```yaml
metric_inverse_max_abs: 1.0e-12
isco_max_abs: 1.0e-12
efficiency_max_abs: 1.0e-10
null_constraint_median_abs: 1.0e-9
null_constraint_p99_abs: 1.0e-7
ray_hit_radius_relative_error: 1.0e-6
disk_luminosity_relative_error: 5.0e-4
spectrum_resolution_relative_l2: 2.0e-3
spectrum_resolution_peak_error: 2.0e-3
injection_recovery_bias_sigma: 0.25
```

These are starting thresholds. Tighten or relax only with written justification.

### Regression data

Commit only small, transparent benchmark files. Large outputs belong in a release archive.

---

# Part V — Simulation and Inference Design

## 16. Fiducial Physical System

Use a stellar-mass black-hole binary as the primary synthetic system because the thin-disk thermal peak lies in the X-ray band.

Initial fiducial parameters:

```yaml
mass_msun: 10.0
distance_kpc: 8.0
inclination_deg: 45.0
spin: 0.7
outer_radius_rg: 1000.0
eddington_ratios: [0.03, 0.06, 0.10, 0.18, 0.30]
f_col_model:
  type: luminosity_law
  f0: 1.7
  f1: 0.10
  pivot: 0.10
  lower: 1.4
  upper: 2.2
inner_stress:
  type: zero_torque
```

These are development values, not final scientific choices.

---

## 17. Production Parameter Grid

### Main grid

Use a design such as:

```text
spin:
[-0.9, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.998]

inclination_deg:
[20, 40, 60, 75]

eddington_ratio:
[0.03, 0.06, 0.10, 0.18, 0.30]

f_col_baseline:
[1.5, 1.7, 1.9, 2.1]

inner_stress_delta_eta:
[0.0, 0.005, 0.01, 0.02]
```

Do not blindly execute the Cartesian product at maximum resolution. Use:

1. coarse screening grid;
2. adaptive refinement near high bias;
3. final confirmatory runs at higher resolution.

### Replicates

For noisy synthetic spectra, use enough independent noise realizations to estimate bias and coverage. Begin with 20 for pipeline testing. Increase to at least 100 per selected condition for final coverage claims, subject to a precision calculation.

### Randomness

Use a hierarchical seed scheme:

```text
master_seed
  -> condition_seed
     -> epoch_seed
        -> noise_seed
```

Store every derived seed.

---

## 18. Synthetic Spectra

### Detector-independent baseline

Use logarithmic energy bins, for example:

\[
0.1\ {\rm keV}\le E\le20\ {\rm keV}.
\]

Store:

- energy-bin edges;
- energy-bin centers;
- photon flux;
- energy flux;
- noiseless expectation;
- noisy realization;
- variance model;
- true parameters.

### Noise models

Implement separately:

1. Gaussian relative-error model for debugging.
2. Poisson counts with a simple effective area and exposure.
3. Optional real instrument response.

Do not mix Gaussian and Poisson likelihoods.

### Optional instrument folding

Only after detector-independent validation:

- use a documented public response;
- preserve the response file and checksum;
- state the instrument, mode, exposure, background treatment, and energy filtering;
- do not claim observational realism without background and calibration limitations.

---

## 19. Models to Fit

### Model M0 — Standard restrictive model

- zero torque;
- fixed \(f_{\rm col}=1.7\);
- shared spin;
- per-epoch accretion rate.

### Model M1 — Constant free color correction

- zero torque;
- one shared free \(f_{\rm col}\);
- shared spin;
- per-epoch accretion rate.

### Model M2 — Epochwise color correction

- zero torque;
- separate \(f_{\rm col}\) per epoch;
- shared spin;
- per-epoch accretion rate.

### Model M3 — Luminosity-dependent color correction

- zero torque;
- \(f_{\rm col}(\ell)\) law;
- shared spin;
- per-epoch accretion rate.

### Model M4 — Stress-aware model

- nonzero inner-edge stress parameter;
- selected color-correction model;
- shared spin.

M4 is used to test whether the true parameter becomes identifiable when the relevant physics is included.

---

## 20. Statistical Plan

### Primary estimands

For each condition:

\[
\Delta a_\ast = \hat a_\ast-a_{\ast,\rm true},
\]

relative or absolute uncertainty width, interval coverage, and fit quality.

### Required outputs

- median spin bias;
- mean spin bias;
- root-mean-square error;
- 68% and 95% interval coverage;
- posterior width;
- catastrophic-failure rate;
- goodness-of-fit statistic;
- posterior correlation matrix;
- computational cost.

### Sampling

Use:

- deterministic optimizer for initialization;
- nested sampling or ensemble MCMC for final selected cases;
- profile likelihood or Laplace approximation for the broad screening grid only if validated.

### Priors

Write all priors in `paper/tables/priors.csv`. Use physically justified bounds. Never let a prior silently exclude the injected truth.

### Simulation-based calibration

For same-model injection recovery:

1. Draw parameters from the declared prior.
2. Simulate data.
3. Fit data.
4. Compute posterior rank of the true parameter.
5. Check rank uniformity and coverage.

Do not interpret misspecified-model noncoverage as a sampler failure before same-model calibration passes.

### Blinding

Generate the final confirmatory injections with hidden true values stored in a file not read by the analysis script until model choices and thresholds are frozen.

---

# Part VI — Figures and Tables

## 21. Mandatory Figures

### Figure 1 — Model schematic

Kerr black hole, equatorial thin disk, ISCO, observer inclination, image plane, and photon paths. Clearly label model assumptions.

### Figure 2 — Validation of orbital physics

Panels:

- \(r_{\rm ISCO}(a_\ast)\);
- \(\eta_{\rm NT}(a_\ast)\);
- selected \(E,L,\Omega\) profiles;
- numerical minus analytic residuals.

### Figure 3 — Disk flux and temperature

Profiles for representative retrograde, zero, moderate, and high prograde spin. Include convergence residuals in a lower panel or separate figure.

### Figure 4 — Ray-traced disk images

At least two spins and three inclinations, with identical normalization rules stated in the caption.

### Figure 5 — Relativistic spectra

Show effects of:

- spin;
- inclination;
- color correction;
- weak inner stress.

Avoid changing multiple parameters in one comparison unless the purpose is degeneracy.

### Figure 6 — Single-epoch degeneracies

Likelihood/posterior contours for \(a_\ast\), \(f_{\rm col}\), inclination, and accretion rate.

### Figure 7 — Multi-epoch improvement

Compare posterior spin constraints for 1, 2, 3, and 5 epochs.

### Figure 8 — Bias map

Heat map of spin bias versus true spin and inclination, faceted by model mismatch.

### Figure 9 — Coverage map

68% and 95% interval coverage over the main parameter domain.

### Figure 10 — Failure domain

Region where fit quality appears acceptable but spin is materially biased. This can be the strongest scientific figure.

### Figure 11 — Numerical convergence

Resolution, ODE tolerance, outer radius, and energy-grid sensitivity.

---

## 22. Mandatory Tables

1. Model assumptions and included physics.
2. Fiducial parameters and prior ranges.
3. Analytic benchmark values.
4. Numerical convergence summary.
5. Injection-recovery summary.
6. Bias and coverage summary.
7. Comparison with relevant published models/software.
8. Computational cost and hardware.
9. Limitations and expected direction of neglected effects.

All tables must be generated from CSV/Parquet/HDF5 data. No manual number entry into LaTeX.

---

# Part VII — Phase-by-Phase Codex Execution

## 23. Phase 0 — Literature and Scope Freeze

### Tasks

- Build the 40+ paper gap matrix.
- Read the foundational papers and current uncertainty studies.
- Search recent preprints through the project start date.
- Identify existing open-source packages.
- Write a one-page novelty decision.
- Freeze a primary question and two fallback questions.

### Deliverables

```text
literature/gap_matrix.csv
literature/novelty_decision.md
literature/references.bib
docs/assumptions.md
```

### Acceptance criteria

- At least 40 directly relevant works.
- At least 10 works from the previous five years.
- At least 5 directly competing tools/models.
- Exact overlap and difference stated without marketing language.

### Codex prompt

```text
Act as a computational astrophysics research assistant. Build a structured
literature-gap audit for the KerrDisk-UQ project. Do not invent references.
For each source, capture model assumptions, ray tracing, varied parameters,
systematics, inference method, code availability, and overlap with the planned
multi-epoch spin-bias study. Produce gap_matrix.csv, references.bib, and a
cautious novelty_decision.md. Any unverified item must be marked UNVERIFIED.
Do not claim novelty until the matrix supports it.
```

---

## 24. Phase 1 — Repository Bootstrap

### Tasks

- Create repository structure.
- Configure `uv`, `ruff`, `mypy`, `pytest`, pre-commit, and GitHub Actions.
- Create configuration models.
- Add logging and run manifests.
- Add license and contribution files.

### Acceptance criteria

```bash
uv run ruff check .
uv run mypy src/kerrdisk
uv run pytest -q
```

All pass on Linux and Windows CI where practical.

### Codex prompt

```text
Bootstrap the kerrdisk-uq repository exactly as specified in the research
guide. Use a src layout, typed public APIs, Pydantic configuration validation,
structured run manifests, pytest, Ruff, mypy, pre-commit, and GitHub Actions.
Do not implement astrophysics yet. Add smoke tests and document every command.
```

---

## 25. Phase 2 — Metric, ISCO, and Circular Orbits

### Tasks

- Implement Kerr metric and inverse.
- Implement derivatives.
- Implement horizon and ISCO.
- Implement circular orbit quantities.
- Implement efficiency.
- Add unit and property-based tests.

### Acceptance criteria

- All benchmark values pass.
- Metric inverse residual below tolerance.
- No NaNs in the declared valid domain.
- Domain errors are explicit and informative.

### Codex prompt

```text
Implement and test Kerr metric utilities, horizon radius, analytic equatorial
ISCO, circular-orbit angular velocity, specific energy, specific angular
momentum, and Novikov-Thorne efficiency. Follow the sign convention in the
guide. Add equation-source comments referencing equation_registry.md. Include
unit tests for all benchmark values and Hypothesis tests over the valid domain.
Do not implement disk flux until these tests pass.
```

---

## 26. Phase 3 — Page–Thorne Disk Flux

### Tasks

- Register the exact flux equation and convention.
- Implement numerical derivatives and quadrature carefully.
- Implement physical unit conversion.
- Integrate luminosity.
- Add Newtonian-limit and efficiency tests.
- Produce validation plots.

### Acceptance criteria

- Zero flux at the ISCO within numerical tolerance.
- Positive flux outside the inner edge.
- Luminosity normalization passes.
- Resolution convergence passes.

### Codex prompt

```text
Implement the zero-torque relativistic thin-disk flux using the verified
Page-Thorne conservation-law integral recorded in equation_registry.md.
Separate dimensionless geometry from physical units. Add tests for the ISCO
boundary, positivity, Newtonian large-radius scaling, luminosity efficiency,
and radial-grid convergence. Generate validation data but do not generate
publication claims.
```

---

## 27. Phase 4 — Atmosphere and Local Spectrum

### Tasks

- Implement Planck spectrum safely.
- Implement diluted blackbody.
- Implement color-correction model interface.
- Test bolometric normalization.
- Test extreme-energy numerical stability.

### Acceptance criteria

- No overflow/underflow warnings over production energy and temperature ranges.
- Integrated local spectrum agrees with \(\sigma T_{\rm eff}^4/\pi\) under the adopted intensity convention.
- \(f_{\rm col}\) preserves bolometric flux.

### Codex prompt

```text
Implement local thermal emission models for KerrDisk-UQ. Include a stable
Planck function, diluted blackbody color correction, constant and
luminosity-dependent f_col models, and explicit angular-emission interfaces.
Verify bolometric normalization numerically. Keep limb darkening disabled in
the baseline.
```

---

## 28. Phase 5 — Reference Ray Tracer

### Tasks

- Implement observer tetrad.
- Implement null-ray initial conditions.
- Implement scalar Hamiltonian geodesic integration.
- Implement disk, horizon, escape, and failure events.
- Track invariants.
- Produce ray diagnostics.

### Acceptance criteria

- Null constraint and conserved quantities pass.
- Shadow boundary passes an independent benchmark.
- Disk-hit locations converge.
- No silent `MAX_STEPS` rays in production images.

### Codex prompt

```text
Implement a correctness-first scalar backward Kerr ray tracer using the
Hamiltonian equations. Derive initial photon momenta from a documented observer
tetrad at large radius. Add event detection for disk crossing, horizon capture,
escape, maximum steps, and numerical failure. Track all invariants and write
diagnostic tests. Do not optimize until the scalar implementation passes all
validation criteria.
```

---

## 29. Phase 6 — Transfer Map and Observed Spectrum

### Tasks

- Calculate disk intersection data.
- Calculate emitter four-velocity.
- Calculate redshift factor.
- Integrate intensity over the observer screen.
- Cache transfer maps.
- Add distance and symmetry tests.

### Acceptance criteria

- \(D^{-2}\) scaling passes.
- Screen-resolution convergence passes.
- Same transfer map reused across energy grid without physics changes.
- Flux units are documented and tested.

### Codex prompt

```text
Build the transfer-map and observed-spectrum layer on the validated ray tracer.
For each successful disk ray store screen solid angle, emission radius,
azimuth, redshift factor, emission angle, and diagnostics. Use invariance of
I_nu/nu^3 to compute spectra. Add tests for distance scaling, symmetry,
screen-resolution convergence, and cache reproducibility.
```

---

## 30. Phase 7 — Independent Cross-Validation

### Tasks

Cross-check with at least one independent route:

- trusted published tables;
- another open ray-tracing implementation;
- analytic Kerr shadow boundary;
- independent notebook coded from equations without importing production functions.

### Acceptance criteria

- Agreement threshold declared before comparison.
- Disagreement investigated, not averaged away.
- Validation report committed.

### Deliverable

```text
docs/validation.md
data/processed/validation_summary.csv
```

### Codex prompt

```text
Create an independent validation path for KerrDisk-UQ. It must not merely call
the production functions through a different wrapper. Compare ISCO, efficiency,
ray invariants, Kerr shadow boundary, disk flux profiles, and selected spectra.
Write a validation report with declared tolerances, residual plots, and all
unresolved differences.
```

---

## 31. Phase 8 — Controlled Inner-Stress Extension

### Tasks

- Select an established model from the literature.
- Register equations and conventions.
- Implement zero-parameter recovery.
- Validate added efficiency.
- Restrict parameter domain.

### Acceptance criteria

- Exact baseline recovery.
- Correct integrated extra luminosity.
- No negative unphysical flux in the adopted domain.
- Documentation states this is not GRMHD.

### Codex prompt

```text
After identifying and citing an established nonzero-ISCO-stress thin-disk
prescription, implement it as an optional model. Parameterize it by a physically
interpretable added efficiency if supported by the source. Prove numerically
that zero stress recovers the baseline and that the luminosity increment and
asymptotic behavior match the published prescription.
```

---

## 32. Phase 9 — Synthetic Data and Likelihood

### Tasks

- Implement deterministic synthetic spectra.
- Implement Gaussian debug noise and Poisson production noise.
- Implement exact matching likelihoods.
- Add seed manifests.
- Add same-model injection recovery.

### Acceptance criteria

- Noise moments agree with theory.
- True parameters recover in high-signal tests.
- Likelihood is finite across the valid prior.
- Invalid physics returns explicit rejection, not NaN propagation.

### Codex prompt

```text
Implement synthetic multi-epoch datasets and statistically matched
likelihoods. Use deterministic seed derivation and store all truth metadata.
First pass same-model injection recovery with an optimizer, then with the final
sampler. Add tests for noise statistics, likelihood normalization where
applicable, and recovery of injected spin.
```

---

## 33. Phase 10 — Inference and Simulation-Based Calibration

### Tasks

- Implement optimizer.
- Implement sampler adapters.
- Add convergence diagnostics.
- Run prior-predictive checks.
- Run simulation-based calibration.
- Freeze priors.

### Acceptance criteria

- Repeated chains/samplers agree.
- Effective sample size and convergence criteria pass.
- Rank plots do not show major pathology.
- Same-model 68% and 95% coverage is acceptably calibrated.

### Codex prompt

```text
Build the inference layer with a common parameter-transform and likelihood API.
Implement robust optimization followed by emcee or dynesty for final cases.
Run prior predictive checks and simulation-based calibration. Refuse to start
the misspecification science campaign until same-model recovery and coverage
pass the documented thresholds.
```

---

## 34. Phase 11 — Screening Campaign

### Tasks

- Run coarse grid.
- Identify high-bias and low-identifiability regions.
- Profile runtime.
- Refine the final design.
- Freeze the confirmatory analysis protocol.

### Acceptance criteria

- All conditions have status records.
- Failure causes summarized.
- No post-hoc removal without criteria.
- Final grid chosen before confirmatory truths are unblinded.

### Codex prompt

```text
Run the coarse KerrDisk-UQ screening campaign using versioned configurations.
Summarize bias, posterior width, fit quality, failure rates, and runtime.
Identify regions requiring refinement. Do not write final conclusions. Produce
a frozen confirmatory protocol and machine-readable condition list.
```

---

## 35. Phase 12 — Confirmatory Campaign

### Tasks

- Run high-resolution selected conditions.
- Use hidden true values where possible.
- Use enough replicates for coverage.
- Perform numerical-resolution reruns.
- Unlock truth only after analysis freeze.

### Acceptance criteria

- Planned replicate count completed or transparently reported.
- Convergence failures included in the accounting.
- Bias conclusions stable under higher numerical resolution.
- Main claims map directly to predeclared estimands.

### Codex prompt

```text
Execute the frozen confirmatory campaign without changing models, priors,
thresholds, or plotting scales. Use the locked condition list and hidden truths.
After completion, compute the predeclared bias, RMSE, coverage, fit-quality, and
failure metrics. Then unblind and produce the final machine-readable results.
```

---

## 36. Phase 13 — Figures, Tables, and Claim Audit

### Tasks

- Generate all figures from scripts.
- Generate all tables from scripts.
- Create `paper/claim_audit.md`.

Each claim-audit row must contain:

```text
claim_id,manuscript_location,claim_text,evidence_file,figure_or_table,
analysis_script,assumptions,limitations,status
```

### Acceptance criteria

- Every numerical claim has an evidence file.
- Figure captions state normalization and varied parameters.
- No causal language for a purely sensitivity-based result.
- Failed hypotheses are reported.

### Codex prompt

```text
Generate publication figures and tables exclusively from final versioned data.
Create a claim audit linking every quantitative statement to its configuration,
output file, analysis script, and figure/table. Flag unsupported or overstated
claims. Do not improve visual appearance by hiding outliers or failures.
```

---

## 37. Phase 14 — Manuscript

### Recommended paper structure

1. **Introduction**
   - astrophysical importance of spin;
   - continuum-fitting principle;
   - known systematics;
   - exact unresolved question;
   - contribution summary.

2. **Physical Model**
   - Kerr geometry;
   - ISCO and circular orbits;
   - Page–Thorne disk;
   - atmosphere/color correction;
   - optional inner stress;
   - scope and validity.

3. **Numerical Methods**
   - disk flux;
   - ray tracing;
   - transfer maps;
   - synthetic data;
   - inference;
   - resolution.

4. **Validation**
   - analytic checks;
   - independent checks;
   - convergence;
   - injection recovery.

5. **Experiment Design**
   - parameters;
   - priors;
   - multi-epoch design;
   - misspecification scenarios;
   - metrics.

6. **Results**
   - baseline spectra;
   - degeneracies;
   - bias maps;
   - multi-epoch effects;
   - coverage;
   - failure domain.

7. **Discussion**
   - interpretation;
   - relation to earlier studies;
   - implications for continuum fitting;
   - limitations;
   - future observational application.

8. **Conclusions**

9. **Data and Software Availability**

10. **AI Assistance Disclosure**, if required by the journal or included voluntarily.

### Writing constraints

- Do not say “realistic” without defining what is realistic.
- Do not call synthetic data “experimental.”
- Do not call a numerical curve “evidence” for an astrophysical population.
- Do not infer observational spin errors from ideal detector-free spectra without qualification.
- State that the simulator is a thin-disk model, not a GRMHD simulation.
- Use exact dates and versions for software and archived releases.

### Codex prompt

```text
Draft the manuscript from the claim audit and final outputs. Every quantitative
statement must cite a figure, table, or machine-readable result. Separate
validation from new results. Compare directly with the literature gap matrix.
Do not claim novelty, accuracy, realism, or observational impact beyond the
evidence. Include data/software availability and a transparent Codex/AI usage
statement.
```

---

# Part VIII — Reproducibility and Release

## 38. Reproduction Command

The complete paper should be reproducible with:

```bash
uv sync --all-extras
uv run python scripts/reproduce_all.py --config configs/production/release.yaml
```

For expensive ray tracing, support two modes:

```bash
# Uses archived validated transfer maps
uv run python scripts/reproduce_all.py --mode archive

# Recomputes all transfer maps
uv run python scripts/reproduce_all.py --mode full
```

The archive mode must reproduce every paper figure and table.

---

## 39. Release Checklist

- [ ] Tests pass from a clean clone.
- [ ] Documentation builds.
- [ ] Tutorial runs.
- [ ] All paper figures regenerate.
- [ ] `CITATION.cff` is valid.
- [ ] License is OSI approved if pursuing a software paper.
- [ ] Version tag created.
- [ ] GitHub release created.
- [ ] Zenodo or equivalent archive created with DOI.
- [ ] DOI added to `CITATION.cff`.
- [ ] Data archive has checksums and metadata.
- [ ] Environment lockfile committed.
- [ ] `AI_USAGE.md` complete.
- [ ] Known limitations and numerical failures documented.
- [ ] No private credentials, copyrighted response files, or restricted data committed.

---

## 40. Data Availability Template

Adapt to the actual archive:

```text
The source code, configuration files, validation tests, processed simulation
outputs, and scripts required to reproduce the figures and tables are archived
in [repository name and DOI]. Large intermediate ray-tracing products are
provided as validated transfer-map archives. The repository records the
software environment, random seeds, run manifests, and checksums used for the
published analysis.
```

Do not write this until the DOI exists.

---

## 41. AI Usage Template

Adapt to the target journal:

```text
OpenAI Codex was used to assist with code drafting, refactoring, test
generation, documentation, and language editing. All physical equations were
checked against cited sources. All generated code was reviewed by the authors
and was subjected to analytic tests, independent numerical comparisons,
convergence studies, and reproducible execution. Codex did not determine the
scientific conclusions independently.
```

Never claim “no AI was used” when Codex was used.

---

# Part IX — Journal Strategy

## 42. Primary Science-Paper Route

Potential targets depend on final novelty and rigor:

- **MNRAS** — appropriate for a substantive astrophysical methods/results paper.
- **The Astrophysical Journal** — appropriate if the astrophysical question, validation, and inference are strong.
- **Astronomy & Astrophysics** — appropriate if the result has broad astrophysical relevance and the methods are mature.
- **New Astronomy** or another reputable specialist journal — possible fallback if the contribution is narrower.

Verify current scope, article type, templates, code/data policy, and publication charges immediately before submission.

### Recommended decision rule

Submit to a major astrophysics journal only when:

- the novelty gate passes;
- full relativistic transfer is validated;
- the uncertainty study is broad enough for general conclusions;
- the conclusions survive convergence tests;
- the paper explains why the findings matter for real spin inference.

---

## 43. Separate Software-Paper Route

A software paper is optional and should not duplicate the science paper.

A JOSS-style submission requires a mature open-source package, documentation, tests, an obvious research application, and a concise software-focused paper. The software paper should not present the new science result as its focus.

Possible sequence:

1. Publish the astrophysical science paper.
2. Obtain external use, benchmarks, or adoption evidence.
3. Submit a separate software paper describing the package and research need.

Do not split one weak project into two papers merely to increase paper count.

---

# Part X — Submission Package

## 44. Before Submission

- [ ] Read the target journal’s current author instructions.
- [ ] Download the current LaTeX template.
- [ ] Confirm word/page/figure requirements.
- [ ] Confirm data availability requirements.
- [ ] Confirm software citation requirements.
- [ ] Confirm AI disclosure requirements.
- [ ] Check every reference through ADS/DOI.
- [ ] Run plagiarism/similarity checks ethically.
- [ ] Run a reference completeness audit.
- [ ] Verify author contributions.
- [ ] Verify affiliations and ORCID records.
- [ ] Verify figure fonts and color accessibility.
- [ ] Test the archive DOI.
- [ ] Compile from a clean repository clone.
- [ ] Ask an astrophysicist familiar with accretion disks to review the physics.

### External expert review is strongly recommended

Codex cannot replace expert scientific review. Before submission, obtain feedback from someone who can evaluate relativistic accretion-disk physics and continuum-fitting assumptions.

---

## 45. Cover Letter Structure

The cover letter should state:

1. manuscript title;
2. exact research question;
3. two or three principal findings;
4. why they matter for the journal audience;
5. exact novelty relative to the closest papers;
6. open code/data archive;
7. originality and exclusive-submission statements;
8. optional suggested reviewers and exclusions, following journal rules.

Do not write exaggerated claims such as “revolutionary,” “first ever,” or “proves.”

---

## 46. Responding to Reviewers

Create a point-by-point response:

```text
Reviewer comment:
[verbatim comment]

Response:
[respectful answer]

Change made:
[exact manuscript section, page, line, figure, or table]

Evidence:
[test, analysis, citation, or new result]
```

When a reviewer identifies a real limitation, revise the claim rather than attempting to “win” the argument.

---

# Part XI — Risk Register

## 47. Major Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Generic research question | Desk rejection | Complete novelty gate and use a precise inference question |
| Incorrect Page–Thorne normalization | Entire result invalid | Equation registry, dimensional analysis, luminosity-efficiency test |
| Incorrect ray initial conditions | Wrong images and spectra | Tetrad derivation, invariant tests, shadow benchmark |
| Near-horizon ODE instability | Biased high-spin results | Adaptive solver, event handling, coordinate/tolerance study |
| Underresolved observer screen | Spectral bias | Adaptive screen and convergence analysis |
| Fixed \(f_{\rm col}\) presented as atmosphere physics | Overclaim | Call it a sensitivity model; compare several laws |
| Inner torque model invented ad hoc | Weak physical meaning | Use a cited model and integrated-efficiency validation |
| Too many free parameters | Non-identifiable inference | Staged models, priors, multi-epoch sharing, synthetic calibration |
| High S/N hides model error | False confidence | Coverage and misspecification tests |
| Codex generates plausible but wrong equations | Scientific failure | Source registry and independent human review |
| Runtime becomes excessive | Incomplete campaign | Transfer-map caching, screening grid, surrogate only after validation |
| JOSS rejection for immature software | Lost effort | Seek external use and follow current review criteria |
| AI disclosure omitted | Ethics/policy problem | Maintain `AI_USAGE.md` from day one |

---

# Part XII — Final Definition of Done

The project is complete only when all boxes below are checked.

## Physics

- [ ] Kerr metric, ISCO, orbit, and efficiency tests pass.
- [ ] Page–Thorne flux normalization passes.
- [ ] Ray invariants and shadow benchmark pass.
- [ ] Spectrum convergence passes.
- [ ] Same-model injection recovery passes.
- [ ] Misspecification study completed.
- [ ] Main results stable under numerical refinement.

## Research

- [ ] Literature gap documented.
- [ ] Exact novelty statement defensible.
- [ ] Hypotheses recorded before confirmatory runs.
- [ ] Bias and coverage maps completed.
- [ ] Limitations stated clearly.
- [ ] Independent astrophysics review completed.

## Software

- [ ] Clean install succeeds.
- [ ] Tests and CI pass.
- [ ] Documentation and tutorial complete.
- [ ] Open license and citation metadata present.
- [ ] Release and archival DOI created.
- [ ] Reproduction command works.

## Paper

- [ ] Every claim linked to evidence.
- [ ] Figures and tables generated automatically.
- [ ] References verified.
- [ ] Data/software availability accurate.
- [ ] AI usage disclosed accurately.
- [ ] Journal instructions checked on submission date.
- [ ] Cover letter and reviewer-response template ready.

---

# Part XIII — Master Codex Prompt

Use this only after placing this guide in the repository.

```text
You are working inside the KerrDisk-UQ computational astrophysics repository.
Read KERR_DISK_CODEX_RESEARCH_GUIDE.md, AGENTS.md, docs/assumptions.md, and
docs/equation_registry.md before making changes.

Execute only the currently assigned phase. Do not skip phase acceptance
criteria. Never invent equations, citations, benchmark values, simulation
outputs, or manuscript claims. Every physical equation must have a source and
convention in equation_registry.md. Every numerical result must come from a
versioned configuration and output manifest.

Before coding:
1. summarize the requested phase;
2. list files to create or modify;
3. identify scientific risks;
4. identify required validation.

After coding:
1. run formatting, linting, typing, and tests;
2. report exact commands and outcomes;
3. report unresolved warnings or scientific uncertainties;
4. update documentation and the phase checklist;
5. stop if acceptance criteria do not pass.

Generated code is provisional until it passes analytic, convergence, and
independent validation. Do not begin the next phase automatically.
```

---

# Part XIV — Foundational Reading List

Verify all bibliographic fields in NASA ADS or the publisher before submission.

1. Bardeen, J. M., Press, W. H., & Teukolsky, S. A. (1972), *Rotating Black Holes: Locally Nonrotating Frames, Energy Extraction, and Scalar Synchrotron Radiation*, The Astrophysical Journal, 178, 347. DOI: 10.1086/151796.
2. Novikov, I. D., & Thorne, K. S. (1973), *Astrophysics of Black Holes*, in *Black Holes*, ed. C. DeWitt & B. DeWitt.
3. Page, D. N., & Thorne, K. S. (1974), *Disk-Accretion onto a Black Hole. I. Time-Averaged Structure of Accretion Disk*, The Astrophysical Journal, 191, 499. DOI: 10.1086/152990.
4. Thorne, K. S. (1974), *Disk-Accretion onto a Black Hole. II. Evolution of the Hole*, The Astrophysical Journal, 191, 507.
5. Cunningham, C. T. (1975), *The Effects of Redshifts and Focusing on the Spectrum of an Accretion Disk around a Kerr Black Hole*, The Astrophysical Journal, 202, 788.
6. Agol, E., & Krolik, J. H. (2000), *Magnetic Stress at the Marginally Stable Orbit: Altered Disk Structure, Radiation, and Black Hole Spin Evolution*, The Astrophysical Journal, 528, 161. DOI: 10.1086/308177.
7. Kulkarni, A. K., et al. (2011), *Measuring Black Hole Spin by the Continuum-Fitting Method: Effect of Deviations from the Novikov–Thorne Disc Model*, MNRAS, 414, 1183.
8. Penna, R. F., et al. (2010), *Simulations of Magnetized Disks Around Black Holes: Effects of Black Hole Spin, Disk Thickness, and Magnetic Field Geometry*.
9. McClintock, J. E., Narayan, R., & Steiner, J. F. (2014), *Black Hole Spin via Continuum Fitting and the Role of Spin in Powering Transient Jets*.
10. Salvesen, G., & Miller, J. M. (2021), *Black Hole Spin in X-ray Binaries: Giving Uncertainties an \(f\)*.
11. Zhou, M., et al. (2020), *Thermal Spectra of Thin Accretion Disks of Finite Thickness Around Kerr Black Holes*.
12. Abramowicz, M. A., & Fragile, P. C. (2013), *Foundations of Black Hole Accretion Disk Theory*.

---

# Part XV — First Practical Action

Do not begin by asking Codex to “make the whole paper.”

Start with this:

```text
Create only Phase 0 of the KerrDisk-UQ project. Build the literature gap matrix,
verify the foundational references, identify the closest competing papers and
software, and write a cautious novelty decision. Do not implement code. Mark
all uncertain bibliographic or scientific details as UNVERIFIED. The output
must make a go/no-go recommendation for the proposed multi-epoch Kerr spin
bias study and list the exact changes needed if the question is already
covered.
```

After Phase 0 is reviewed by the human author, proceed to Phase 1.

---

## Final Research Principle

A successful paper is not “Python produced attractive images of a Kerr disk.”  
A successful paper is:

> A validated numerical experiment that answers a precise astrophysical question, quantifies uncertainty and failure modes, is reproducible, and makes claims no stronger than the evidence.
