# AI Usage Log

Date: 2026-06-18

OpenAI Codex was used to read the KerrDisk-UQ research guide and create the provisional Phase 0 literature-scope deliverables:

- `literature/gap_matrix.csv`
- `literature/references.bib`
- `literature/novelty_decision.md`
- `docs/assumptions.md`

Codex used web search and available source snippets/pages to identify relevant foundational papers, recent studies, and competing tools/models. Bibliographic or scientific details that were not fully verified were marked `UNVERIFIED`. These outputs require human review and ADS/DOI verification before manuscript use.

No scientific results were generated. No physics code was implemented.

## 2026-06-18 Phase 1 Bootstrap

OpenAI Codex was used to create the initial repository scaffolding, including
metadata, development configuration, GitHub Actions, documentation placeholders,
typed Pydantic configuration models, run-manifest helpers, CLI scaffolding, and
smoke tests.

No Kerr metric, ISCO, disk-flux, ray-tracing, spectrum, synthetic-data,
likelihood, or inference physics was implemented.

## 2026-06-18 Phase 2 Metric, ISCO, and Circular Orbits

OpenAI Codex was used to update the equation registry, implement Kerr metric
utilities, horizon radius, equatorial ISCO radius, circular-orbit angular
velocity, specific energy, specific angular momentum, and the zero-torque
Novikov-Thorne efficiency proxy `1 - E_ISCO`.

Codex also generated unit and property tests for benchmark values, metric
inverse residuals, Schwarzschild limits, and domain errors. No Page-Thorne disk
flux, atmosphere, ray-tracing, spectrum, synthetic-data, likelihood, inference,
or manuscript-result code was implemented.

## 2026-06-18 Phase 3 Page-Thorne Disk Flux

OpenAI Codex was used to verify the Page and Thorne source convention, update
the equation registry, implement the zero-torque Page-Thorne flux integral,
luminosity integrals, physical unit conversion, validation tests, and a
validation artifact command.

The generated validation CSV and PNG are for numerical validation only. No
atmosphere, ray-tracing, spectrum, synthetic-data, likelihood, inference, or
manuscript-result code was implemented.

## 2026-06-18 Phase 4 Atmosphere and Local Spectrum

OpenAI Codex was used to update the equation registry, implement effective
temperature, stable Planck spectral radiance, diluted blackbody emission,
isotropic angular emission, and constant/epochwise/luminosity-law
color-correction models.

Codex also generated tests for bolometric normalization, spectral-hardening flux
preservation, numerical stability in the Wien tail, and color-correction model
validation. No ray-tracing, observed-spectrum, synthetic-data, likelihood,
inference, or manuscript-result code was implemented.

## 2026-06-18 Phase 5 Reference Ray Tracer

OpenAI Codex was used to update the equation registry, implement scalar
Hamiltonian null-geodesic utilities, a static-observer tetrad, documented
large-radius screen initialization, explicit ray outcomes, and invariant
diagnostics.

Codex also generated tests for null constraints, tetrad orthonormality,
conserved quantities, Schwarzschild shadow bracketing, disk-hit events,
`MAX_STEPS`, and `NUMERICAL_FAILURE` accounting. No transfer-map, redshift,
observed-spectrum, synthetic-data, likelihood, inference, or manuscript-result
code was implemented.

## 2026-06-18 Phase 6 Transfer Map and Observed Spectrum

OpenAI Codex was used to update the equation registry, implement circular
emitter four-velocity, measured-frequency and redshift helpers, transfer-map
construction, observed flux-density integration using `I_nu / nu^3` invariance,
and compressed transfer-map cache I/O.

Codex also generated tests for redshift sanity checks, `D^-2` scaling, screen
symmetry, screen-resolution convergence, cache reproducibility, and transfer-map
reuse across frequency grids. No synthetic-data, likelihood, inference, or
manuscript-result code was implemented.

## 2026-06-19 Phase 7 Independent Cross-Validation

OpenAI Codex was used to create an independent validation path, including
separately coded ISCO, efficiency, Schwarzschild radial-ray, shadow-bracket,
Page-Thorne flux, and constant-intensity spectrum checks. Codex also added a
CSV writer and validation residual plot generation.

An initial Page-Thorne flux comparison failed at 400 radial grid points. Codex
investigated the residual instead of suppressing it, found it was localized near
the ISCO and reduced under grid refinement, then recorded the finding in
`docs/validation.md` and `data/processed/validation_summary.csv`.

No inner-stress, synthetic-data, likelihood, inference, or manuscript-result
code was implemented.

## 2026-06-19 Phase 8 Controlled Inner-Edge Stress

OpenAI Codex was used to verify the Agol and Krolik 2000 inner-edge stress
flux convention, update the equation registry, implement the additive
Agol-Krolik stress flux term, enforce a controlled `Delta_eta` domain, and add
tests for zero-stress recovery, nonnegative added flux, added-efficiency
normalization, large-radius scaling, and invalid parameter rejection.

This phase implements a controlled time-steady thin-disk stress prescription.
It does not implement GRMHD, returning radiation, photon capture,
plunging-region emission, synthetic data, likelihoods, inference, or
manuscript-result code.

## 2026-06-19 Phase 9 Synthetic Data and Likelihood

OpenAI Codex was used to update the equation registry, implement deterministic
detector-independent synthetic spectra, hierarchical seed manifests, Gaussian
relative-error debug noise, Poisson count noise with scalar effective area and
exposure, matched Gaussian and Poisson log-likelihoods, and a deterministic
bounded scalar likelihood maximizer for same-model high-signal recovery tests.

Codex also generated tests for seed replay, Gaussian and Poisson noise moments,
manual likelihood normalization formulas, invalid-model rejection, finite
likelihoods over a valid scalar grid, and injected scalar recovery.

No production sampler, posterior coverage claim, simulation-based calibration,
instrument response, background model, or manuscript-result code was
implemented.

## 2026-06-19 Phase 10 Inference and Simulation-Based Calibration

OpenAI Codex was used to update the equation registry, implement bounded
uniform priors, a common posterior API, deterministic coordinate-pattern
optimization, an internal random-walk Metropolis validation adapter,
split-chain R-hat, effective sample size, prior-predictive checks, and
one-dimensional simulation-based calibration summaries.

Codex also generated `paper/tables/priors.csv` as the frozen Phase 10
development prior table and added tests for prior transforms, optimizer
recovery, repeated-chain diagnostics, invalid-control rejection,
prior-predictive output validation, analytic SBC rank/coverage behavior, and
prior-table bounds.

No Phase 11 screening campaign, final production sampler selection,
relativistic-model SBC campaign, misspecification result, or manuscript-result
code was implemented.

## 2026-06-19 Phase 11 Screening Campaign

OpenAI Codex was used to update the equation registry, implement the Phase 11
screening campaign module, add a versioned screening configuration, update
`scripts/run_grid.py`, generate machine-readable condition/status/replicate
artifacts, and write a frozen confirmatory protocol.

The default screening run completed 64 proxy-screening conditions and accounted
for 1280 planned replicates. Codex generated tests for deterministic condition
construction, status accounting, failure retention, refinement flags, config
loading, and output writing.

These Phase 11 outputs are proxy screening artifacts used to validate workflow
and refinement selection. They are not Kerr thin-disk science results, final
posterior coverage claims, Phase 12 confirmatory outputs, or manuscript-result
code.

## 2026-06-19 Phase 12 Confirmatory Campaign

OpenAI Codex was used to update the equation registry, implement the Phase 12
confirmatory campaign module, add a versioned confirmatory configuration,
create `scripts/run_confirmatory.py`, generate blinded condition files,
separate hidden-truth files, an analysis-freeze record, unblinded summary
tables, resolution-rerun comparisons, and failure-accounting artifacts.

The default run consumed the locked Phase 11 condition list, ran 48 conditions
with 100 base replicates each, and recorded higher-resolution spin-grid reruns.
All 48 conditions completed and passed the declared bias-stability tolerance.

Codex then replaced the Phase 11 proxy backend with the registered
`kerr_thin_disk` backend, reran the locked Phase 12 campaign under
`phase12_kerr_thin_disk_v2`, and regenerated the Phase 12 artifacts. The
corrected run completed 48 conditions with 100 base replicates each and passed
the declared higher-resolution bias-stability tolerance for all conditions.

The corrected backend uses Page-Thorne/Agol-Krolik disk flux and
diluted-blackbody local emission. It is not a full image-plane ray-traced
transfer-map campaign.

Codex then added the `ray_traced_transfer` backend before Phase 13, wiring the
Phase 6 scalar transfer-map ray tracer into the Phase 12 confirmatory model.
The v3 backend traced image-plane screen rays, used transfer-map redshift
factors and solid angles, and evaluated the same Page-Thorne/Agol-Krolik plus
diluted-blackbody local emission over disk-hit records.

Codex then upgraded the run to `phase12_ray_traced_transfer_v4`, replacing the
placeholder emission-angle record with a physical `emission_mu`, adding a
normalized electron-scattering limb-darkening option, correcting the screen
quadrature to use cell-centered image-plane rays, and increasing the production
screen to 5x5.

The corrected v4 run completed 48 locked conditions with 100 base replicates
each. The final configured 81/121 spin-grid comparison passed the declared
higher-resolution bias-stability tolerance for all 48 conditions. Codex also
generated Phase 12.5 transfer-validation artifacts for 3x3/5x5/7x7 screen
convergence, photon-capture and returning-radiation diagnostics, and external
comparison status. The external comparison is currently recorded as `SKIPPED`
because no external ray-tracer transfer-map CSV was supplied in the local
workspace.

## 2026-06-19 Phase 13 Figures, Tables, and Claim Audit

OpenAI Codex was used to implement `scripts/make_figures.py` and
`scripts/make_tables.py`, generate the 11 current paper figure files, generate
summary CSV tables, write `paper/figure_captions.md`, and create
`paper/claim_audit.md`.

The claim audit links supported quantitative statements to final evidence
files and explicitly marks unsupported claims. In the Phase 13 audit,
external ray-tracer agreement and joint multi-epoch improvement were marked
`UNSUPPORTED` because the final outputs did not establish those claims.

## 2026-06-19 Phase 13.5 Pre-Manuscript Readiness

OpenAI Codex was used to implement a fixed-luminosity joint two-epoch versus
separate single-epoch comparison before manuscript drafting. Codex added
`src/kerrdisk/multi_epoch.py`, `scripts/run_multi_epoch.py`, a versioned
`configs/production/phase13p5_multi_epoch.yaml`, regenerated Figure 7, added
`paper/tables/table6_multi_epoch_comparison.csv`, and updated the claim audit.

The run completed 24 two-epoch groups and 2400 planned replicates. The result
is condition-dependent: 15/24 groups reduced the mean 68% spin-interval width
under joint fitting, while only 3/24 groups reduced mean absolute spin bias.
Codex therefore did not claim that joint fitting generally improves spin
recovery.

Codex also implemented `scripts/reproduce_all.py` and release metadata
generation. Archive reproduction regenerated all paper tables and figures and
wrote `data/processed/release/run_manifest.json` plus
`data/processed/release/checksums_sha256.csv`. External ray-tracer agreement
remains unsupported because no external benchmark CSV or local external
ray-tracer executable was available.

## 2026-06-19 External Validation and Bibliography Audits

OpenAI Codex was used to add an external ray-tracer backend availability audit
and a Crossref-backed bibliography verification script. The audit recorded no
installed usable external GYOTO, RAPTOR, grtrans, kgeo, or EinsteinPy backend,
so external ray-tracer agreement remains unsupported. The bibliography report
machine-verified 41 of 57 entries and marked 16 entries as requiring manual
review before manuscript use.

## 2026-06-19 Phase 14 Manuscript Draft

OpenAI Codex was used to draft `paper/main.tex` for an APL Computational
Physics-oriented manuscript and to create `paper/references.bib` containing
only cited references that were machine-verified in the bibliography audit.
The draft restricts quantitative statements to versioned tables, figures,
processed outputs, and `paper/claim_audit.md`. Unsupported external
ray-tracer agreement, detector-response claims, high-resolution morphology,
and returning-radiation or photon-capture spectral corrections were excluded.
