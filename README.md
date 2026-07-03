# KerrDisk-UQ

**Validated, reproducible ray-traced Kerr thin-disk spectra and multi-epoch black-hole spin-inference experiments.**

KerrDisk-UQ is a computational-astrophysics framework for asking a precise,
falsifiable question about continuum-fitting black-hole spin measurements:

> Under controlled spectral-hardening (color-correction) and weak inner-edge
> stress misspecification, how biased is a recovered Kerr spin from thermal
> continuum fitting, and can jointly fitting multiple accretion states reduce
> that bias or its uncertainty?

The project is deliberately built as a *validated numerical experiment*, not a
gallery of black-hole images. Every production equation is registered to a
source, every result is reproducible from a versioned configuration and a single
command, and the manuscript claims are held to no more than the evidence
supports.

- **Language / tooling:** Python ≥ 3.12, managed with [`uv`](https://docs.astral.sh/uv/)
- **Quality gate:** 330 tests passing; `ruff`, `ruff format`, and strict `mypy` clean
- **License:** MIT
- **Status:** Validated pipeline with a first *reduced-resolution* v5 science run.
  This is a research codebase, not a submitted paper — see
  [Status & honest limitations](#status--honest-limitations).

---

## What this repository contains

A complete, tested pipeline from Kerr geometry to a marginalized spin posterior:

1. **Relativistic disk physics** — Kerr metric, horizon, ISCO, circular-orbit
   energy/angular-momentum/efficiency, and the zero-torque Page–Thorne surface
   flux, with a controlled Agol–Krolik nonzero-stress extension.
2. **A correctness-first Kerr ray tracer** — Hamiltonian null geodesics with an
   adaptive Dormand–Prince integrator, event detection (disk hit / horizon
   capture / escape), and image-plane transfer maps.
3. **Physically normalized spectra** — a diluted blackbody atmosphere tied to a
   real system (mass, distance, Eddington ratio → `r_g`, `L_Edd`, `Mdot`,
   `T_eff`), producing observed flux in `erg s⁻¹ cm⁻²`.
4. **Bayesian inference** — synthetic data, matched likelihoods, an
   affine-invariant ensemble sampler (`emcee`), a spectral-grid emulator that
   makes the ray-traced model affordable inside a sampler, and simulation-based
   calibration on the actual forward model.
5. **Campaign layer** — screening, a joint marginalized-fit confirmatory
   campaign, and a multi-epoch comparison, all writing machine-readable
   outputs that drive the paper figures and tables.

---

## Validation: why the numbers should be trusted

Validation proceeds bottom-up, with declared tolerances recorded in
[`docs/validation.md`](docs/validation.md):

| Level | Check | Result |
|---|---|---|
| Algebra | Kerr ISCO and Novikov–Thorne efficiency benchmark tables | pass to `1e-12` / `1e-10` |
| Algebra | Metric inverse identity `gᵤₐ gᵃⁿ = δᵤⁿ` | pass to `< 1e-13` |
| Derivatives | Analytic vs finite-difference metric derivatives, and the exact identity `∂(g⁻¹) = −g⁻¹ (∂g) g⁻¹` | pass across the domain |
| Geodesics | Null constraint, energy, angular momentum, Carter constant conservation | conserved to `~1e-11` |
| Ray tracer | Schwarzschild shadow boundary `√27`; adaptive vs fixed-step hit radius | bracketed; agree to `< 1e-4 rₘ` |
| Disk flux | `F(r_ISCO)=0`, Newtonian `r⁻³` limit, luminosity ↔ efficiency | pass |
| Spectra | **Independent analytic Schwarzschild face-on benchmark** vs the full ray tracer | **~4% relative-L1 (PASS)** |
| Convergence | Full-disk screen resolution vs a 112×112 reference | 1.9% (24²), 0.73% (40²), 0.38% (64²) |
| Inference | Same-model injection recovery and simulation-based calibration on the real forward model | recovers injected parameters; calibrated |

The **independent analytic cross-validation** ([`src/kerrdisk/analytic_validation.py`](src/kerrdisk/analytic_validation.py))
reproduces the ray tracer's redshift factor, solid-angle normalization, and flux
integration from a closed-form calculation coded independently of the production
path — the strongest self-contained check in the repository.

External agreement against a third-party ray tracer (GYOTO/RAPTOR/grtrans/kgeo)
is **not yet established** and is documented as such.

---

## Key results (first reduced-resolution v5 run)

From the joint marginalized-fit campaign (48 locked conditions, 30 replicates,
fiducial 10 M⊙ black hole at 8 kpc):

- **Marginalizing the color correction collapses the spin bias.** Condition-mean
  spin bias is modest (range ≈ −0.26 to +0.41, mean ≈ 0.04) — versus the old
  fixed-color one-dimensional fit, where it reached ≈ 1.1.
- **Inner-stress misspecification is dangerous but detectable.** Conditions with
  neglected inner-edge stress stay biased with *narrow* intervals and *elevated*
  χ²/dof (≈ 6–7); color-correction-only cases recover spin with honest, wider
  intervals.
- **Multi-epoch fitting helps some conditions, not all.** Joint two-epoch fitting
  reduced the mean 68% spin-interval width in 12 of 24 groups; the change in bias
  is condition-dependent.

> These are results from a deliberately reduced-resolution first run (24×24
> full-disk screen, 9×5 emulator nodes, 30 replicates). They demonstrate the
> corrected pipeline and the qualitative science; the exact coverage numbers are
> resolution-limited. See [limitations](#status--honest-limitations).

Figures are in [`paper/figures/`](paper/figures/), generated only from versioned
outputs by [`scripts/make_figures.py`](scripts/make_figures.py); the claim audit
linking every quantitative statement to its evidence file is
[`paper/claim_audit.md`](paper/claim_audit.md).

---

## Quickstart

```bash
# clone, then install (all extras includes the inference dependencies)
uv sync --all-extras

# quality gate
uv run ruff check .
uv run ruff format --check .
uv run mypy src/kerrdisk
uv run pytest -q
```

Minimal usage — a physically normalized ray-traced spectrum:

```python
from math import radians
from kerrdisk.spectrum import build_transfer_map, full_disk_screen_axes
from kerrdisk.scales import observer_distance_rg
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import (
    KerrThinDiskSettings, ray_traced_kerr_thin_disk_energy_flux,
)

a_star, inclination_deg, disk_out = 0.7, 40.0, 80.0
alpha, beta = full_disk_screen_axes(disk_out, screen_size=40)
transfer_map = build_transfer_map(
    a_star, alpha, beta,
    observer_radius=1000.0, observer_theta=radians(inclination_deg),
    disk_outer_radius=disk_out,
    observer_distance=observer_distance_rg(distance_kpc=8.0, mass_msun=10.0),
    max_steps=8000, escape_radius=2000.0,
)
bins = make_log_energy_bins(energy_min_kev=0.1, energy_max_kev=20.0, bin_count=24)
spectrum = ray_traced_kerr_thin_disk_energy_flux(  # erg s^-1 cm^-2 per bin
    transfer_map=transfer_map, a_star=a_star,
    eddington_ratio=0.1, f_col=1.7, delta_eta=0.0,
    energy_bins=bins,
    settings=KerrThinDiskSettings(disk_outer_radius_rg=disk_out,
                                  mass_msun=10.0, distance_kpc=8.0),
    limb_darkening="electron_scattering",
)
```

---

## Reproducing the campaign

The joint marginalized-fit v5 campaign and its figures/tables:

```bash
# 1. confirmatory campaign (builds emulators from ray-traced maps, emcee fits)
uv run python scripts/run_joint_campaign.py

# 2. multi-epoch comparison
uv run python scripts/run_joint_multi_epoch.py

# 3. regenerate figures, tables, and the claim audit from the outputs
uv run python scripts/make_figures.py
uv run python scripts/make_tables.py
```

Configuration lives in [`configs/production/phase12_joint_v5.yaml`](configs/production/phase12_joint_v5.yaml).
The reduced-resolution run finishes in ~30 min on a single core; a
full-resolution run (64×64 screen, denser emulator nodes, 100 replicates) is a
longer job.

---

## Repository layout

```
src/kerrdisk/
  metric.py, geodesics.py, isco.py, circular_orbits.py   # Kerr geometry & orbits
  disk_flux.py, atmosphere.py, scales.py                 # flux, emission, unit scales
  raytrace.py, spectrum.py, thermal_spectrum.py          # tracer, transfer maps, spectra
  synthetic.py, likelihood.py                            # synthetic data & likelihoods
  inference.py, emulator.py, joint_inference.py          # sampling, emulator, joint fit
  screening.py, confirmatory.py, multi_epoch.py          # campaigns (1-D scan)
  joint_campaign.py, joint_multi_epoch.py                # campaigns (joint marginalized)
  analytic_validation.py, transfer_validation.py         # independent cross-checks
docs/          equation_registry.md, assumptions.md, numerical_methods.md, validation.md
configs/       versioned run configurations
scripts/       run_*, make_figures.py, make_tables.py, reproduce_all.py
paper/         figures/, tables/, claim_audit.md, figure_captions.md
tests/         unit / integration / regression / convergence
```

Governance and the phase-by-phase execution contract are in
[`KERR_DISK_CODEX_RESEARCH_GUIDE.md`](KERR_DISK_CODEX_RESEARCH_GUIDE.md) and
[`AGENTS.md`](AGENTS.md).

---

## Numerical methods (highlights)

- **Analytic Kerr metric derivatives** used in the geodesic force term (no
  finite-difference step parameter), cross-checked against finite differences.
- **Adaptive Dormand–Prince RK45** integrator with `rtol`/`atol` error control
  and bisection disk-event refinement — about 12× fewer right-hand-side
  evaluations than the fixed-step reference at matching accuracy.
- **Geometry-sized full-disk image plane** at a large observer radius, so the
  screen captures the disk from the ISCO to the outer edge without clipping.
- **Physical absolute normalization**: the observed flux depends on mass and
  spin (through the ISCO), so it carries spin information beyond spectral shape.
- **Spectral-grid emulator** (KERRBB-style precomputed table) that makes the
  ray-traced model fast enough for a multi-parameter ensemble sampler.

Details: [`docs/numerical_methods.md`](docs/numerical_methods.md),
[`docs/equation_registry.md`](docs/equation_registry.md).

---

## Status & honest limitations

This is an in-progress research codebase. Known limitations, all documented in
the code, docs, and claim audit:

- The published-here v5 results are a **reduced-resolution first run**
  (24×24 screen, 9×5 emulator nodes, 30 replicates). Same-model coverage is
  degraded both by genuine misspecification and by the coarse resolution; a
  full-resolution campaign is required before headline coverage claims.
- **External ray-tracer agreement is not established.** The independent
  *analytic* face-on cross-validation passes, but a comparison against an
  installed third-party GR ray tracer is still pending.
- Spectra are **detector-independent** — no instrument response, background, or
  calibration model.
- Returning radiation and photon-capture reprocessing are **diagnosed but not
  reprocessed** into the confirmatory likelihood.
- Multi-epoch fitting shares spin and color correction but treats epoch
  luminosities as fixed metadata.
- Bibliography verification is incomplete (some entries need manual review).

The manuscript source (`paper/main.tex`) is intentionally empty pending a
rewrite against these v5 outputs.

---

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/kerrdisk
uv run pytest -q
uv run pytest --cov=src/kerrdisk --cov-report=term-missing
```

Physics functions are pure where practical; unit conversion is isolated from
dimensionless geometric-unit calculations; plotting never computes physics; and
manuscript tables are produced only from machine-readable files.

---

## Documentation

- [`docs/equation_registry.md`](docs/equation_registry.md) — every production
  equation with its source, convention, and tests
- [`docs/assumptions.md`](docs/assumptions.md) — scientific scope and validity domain
- [`docs/numerical_methods.md`](docs/numerical_methods.md) — algorithms and conventions
- [`docs/validation.md`](docs/validation.md) — validation results and tolerances
- [`paper/claim_audit.md`](paper/claim_audit.md) — claims linked to evidence
- [`AI_USAGE.md`](AI_USAGE.md) — how AI assistance was used and checked

---

## Citation & license

Released under the [MIT License](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff). If you use this code, please cite it and note
that the results here are from a reduced-resolution validation run.

Author: Tanvir Hassan.
