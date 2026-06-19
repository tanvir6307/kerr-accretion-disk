# KerrDisk-UQ

KerrDisk-UQ is a computational astrophysics project for validated Kerr
thin-disk spectra and uncertainty-quantification experiments.

Current status: Phase 13.5. The repository contains project structure,
configuration models, run-manifest scaffolding, documentation placeholders,
smoke tests, validated Kerr metric / ISCO / circular-orbit utilities, a
validated zero-torque Page-Thorne disk-flux implementation, local
Planck/diluted-blackbody atmosphere models, a scalar reference Kerr ray tracer,
a first transfer-map and observed-flux integration layer, independent
cross-validation artifacts, and a controlled Agol-Krolik inner-edge stress flux
extension. It also contains deterministic detector-independent synthetic
spectra, Gaussian and Poisson noise models, matched likelihoods, seed
manifests, a first high-signal same-model scalar recovery check, bounded
uniform priors, deterministic posterior optimization, a random-walk Metropolis
validation adapter, convergence diagnostics, prior-predictive checks, and
simulation-based calibration summaries. It also contains a Phase 11 coarse
screening runner, generated machine-readable screening artifacts, and a frozen
confirmatory protocol based on proxy screening outputs. It also contains a
Phase 12 confirmatory runner, hidden-truth/blinded-analysis artifacts,
unblinded machine-readable summaries, and numerical-resolution rerun summaries.
The corrected Phase 12 run uses the registered `ray_traced_transfer` backend
rather than the Phase 11 proxy or the earlier inclination-projected thermal
backend. It is an image-plane transfer-map campaign using the scalar reference
ray tracer, physical emission angles, electron-scattering limb darkening,
Page-Thorne/Agol-Krolik disk flux, and diluted-blackbody emission. Phase 12.5
adds transfer-map convergence and capture/returning-radiation diagnostics;
Phase 13 adds generated paper figures, summary tables, figure captions, and a
claim audit. Phase 13.5 adds a joint two-epoch versus separate single-epoch
comparison, release reproduction command, run manifest, and checksum manifest.
External ray-tracer agreement remains unsupported unless a genuine external
transfer-map CSV is supplied and passed. Bibliography verification currently
has a machine-readable Crossref report with unresolved manual-review rows.
Remaining limitations are documented in `docs/validation.md` and
`paper/claim_audit.md`.

Archive reproduction:

```bash
uv run python scripts/reproduce_all.py --config configs/production/release.yaml --mode archive
```

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src/kerrdisk
uv run pytest -q
```

The science phases and acceptance criteria are defined in
`KERR_DISK_CODEX_RESEARCH_GUIDE.md`.
