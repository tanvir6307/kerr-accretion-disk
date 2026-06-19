# Assumptions

Date: 2026-06-18

This document records project assumptions frozen after the provisional Phase 0 audit. It is not a substitute for `docs/equation_registry.md`; no production physical equation should be implemented until its source, convention, normalization, and tests are recorded there.

## Scientific Scope

- The baseline model is a Kerr, geometrically thin, optically thick, equatorial accretion disk.
- The baseline stress condition is zero torque at the ISCO.
- The baseline emission model is a local diluted blackbody with a controlled spectral hardening factor.
- Ray tracing and transfer maps are required before any production-level spectrum is treated as scientifically meaningful.
- Synthetic spectra are detector-independent in Phase 9. Poisson spectra use
  only scalar effective area and exposure until a documented instrument response
  layer is added.
- Gaussian debug spectra and Poisson count spectra must be analyzed with their
  matching likelihoods; they are not interchangeable.
- Instrument response folding, reflection, Comptonization, returning radiation, polarization, tilted disks, and GRMHD evolution are out of scope for the first implementation unless a later phase explicitly adds them with new validation.

## Validity Domain

- The initial luminosity domain is `0.03 <= L/L_Edd <= 0.30`, pending literature justification and numerical checks.
- Runs outside the adopted thin-disk validity domain must be flagged and excluded from primary claims unless a separate model supports them.
- Near-extremal spin cases require special convergence tests because horizon/ISCO proximity can stress numerical integration.

## Inference Scope

- The scientific target is a controlled misspecification study, not a claim that any particular observed black-hole spin is wrong.
- Same-model injection recovery and coverage calibration must pass before misspecified fits are interpreted.
- Multi-epoch fitting should share black-hole spin, mass, distance, and inclination while allowing epoch-level accretion rate and optionally epoch-level or luminosity-dependent `f_col`.
- Priors, estimands, coverage definitions, and failure accounting must be frozen before confirmatory simulations.
- Phase 10 priors are frozen in `paper/tables/priors.csv` for development and
  validation. Any Phase 11 campaign prior change requires updating that table
  and documenting why the previous frozen table was superseded.
- Phase 11 screening artifacts are workflow-screening outputs. They may guide
  Phase 12 refinement selection, but they are not final astrophysical evidence.

## Color-Correction Assumptions

- Constant and luminosity-dependent `f_col` prescriptions are sensitivity models unless their parameters are directly tied to a cited atmosphere calculation.
- No manuscript claim should describe a chosen `f_col(L/L_Edd)` relation as realistic unless that word is explicitly defined and sourced.
- Salvesen and Miller 2021 is a required read before finalizing the `f_col` uncertainty range.

## Inner-Stress Assumptions

- The Phase 8 nonzero inner-edge stress model follows Agol and Krolik 2000 as
  a controlled thin-disk sensitivity prescription.
- A weak ISCO-stress model and an intra-ISCO/plunging-region emission model are
  not interchangeable.
- The selected stress parameterization has a tested zero-stress limit,
  nonnegative added flux over the adopted domain, and an integrated luminosity
  increment that matches `Delta_eta` under the registered convention.
- The controlled production domain is `0 <= Delta_eta <= 0.1`; larger stress
  values require a new source review and validation decision.

## Software/Reproducibility Assumptions

- Every important result must be reproducible from a committed configuration and one command.
- Production runs must not be overwritten.
- Every run must record resolved configuration, git commit, package versions, random seed, timestamps, warnings, convergence status, and checksums.
- Generated code is provisional until it passes analytic checks, independent comparison, convergence tests, and same-model inference checks.
- Phase 10 adds analytic same-model sampler and SBC checks. Full
  relativistic-model posterior coverage and production-sampler calibration are
  not yet validated.
- Phase 11 generated a frozen confirmatory protocol from proxy screening
  outputs. Any condition-list change before Phase 12 must create a new
  versioned protocol.
- Phase 12 was rerun with the `phase12_ray_traced_transfer_v4` backend,
  replacing the Phase 11 proxy and the earlier inclination-projected thermal
  backend with an image-plane transfer-map campaign based on scalar ray
  tracing, physical emission angles, electron-scattering limb darkening,
  Page-Thorne/Agol-Krolik flux, and diluted-blackbody emission.
- The corrected Phase 12 backend supports transfer-map ray-traced claims at
  the configured scalar-reference resolution. Phase 12.5 adds screen
  convergence and capture/returning-radiation diagnostics. It does not support
  claims about iterative returning-radiation reprocessing, flux renormalization
  from photon-capture fractions, independent external ray-tracer agreement, or
  high-resolution image morphology beyond the validated 5x5 transfer maps.
- Phase 13.5 adds a fixed-luminosity joint two-epoch versus separate
  single-epoch comparison. It supports claims that the comparison was run and
  that width/bias changes are condition-dependent. It does not support a broad
  claim that joint multi-epoch fitting improves spin recovery in general.

## Open Items

- Verify all `UNVERIFIED` bibliography fields in `literature/references.bib`.
- Complete full-text review of the closest competing papers before finalizing Phase 1/2 science requirements.
- Choose the final inner-stress/plunging-region prescription only after source review.
- Decide whether retrograde spin is in the first confirmatory grid or reserved for a focused follow-up, based on runtime.
- Supply an external GYOTO/RAPTOR/grtrans-style transfer-map benchmark CSV, or
  install and run an external ray tracer, before claiming external agreement.
