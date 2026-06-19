# Phase 0 Novelty Decision

Date: 2026-06-18

Status: provisional literature-gap audit. This is not yet a claim of novelty. Rows marked `UNVERIFIED` in `gap_matrix.csv` and `references.bib` must be checked against NASA ADS, DOI records, and the final publisher pages before manuscript use.

## Primary Question Under Review

Under what combinations of black-hole spin, inclination, luminosity, spectral-hardening uncertainty, and weak inner-edge stress does a standard zero-torque Novikov-Thorne fit return a biased Kerr spin, and can joint multi-epoch fitting reduce that bias?

## Closest Prior Work

The closest foundations and competitors are:

- Page and Thorne / Novikov and Thorne for the steady relativistic thin-disk flux and zero-torque baseline.
- Cunningham for relativistic transfer functions.
- KERRBB and BHSPEC for established relativistic continuum-fitting models.
- Kulkarni et al. 2011 and Penna et al. 2010 for GRMHD tests of Novikov-Thorne assumptions and continuum-fitting bias from simulated disks.
- Agol and Krolik 2000, Krolik 1999, Gammie 1999, and later GRMHD work for nonzero stress and plunging-region motivation.
- Salvesen and Miller 2021 for the direct warning that spectral hardening uncertainty can dominate continuum-fitting spin errors.
- Mummery et al. 2024/2025 for recent plunging-region emission and spin/stress degeneracy in real systems.
- Recent Cyg X-1, GRS 1716-249, MAXI J0637-430, and MAXI J1820+070 papers for current observational model-dependence.
- Existing software/model ecosystems: KERRBB/KERRBB2, BHSPEC, SLIMBH, NKBB/RayTransfer, KYN/KY, GYOTO, grtrans/geokerr, RAPTOR, Odyssey.

## Direct Answers Required by the Guide

Has joint multi-epoch spin recovery under simultaneous color-correction and inner-stress misspecification already been mapped?

Provisional answer: no exact match found in this first audit. The ingredients have been studied separately or in source-specific ways: color-correction uncertainty is emphasized by Salvesen and Miller 2021; GRMHD/NT deviations are studied by Kulkarni et al. 2011 and Penna et al. 2010; plunging-region/ISCO-stress alternatives are active in Mummery et al. 2024/2025. I did not find a predeclared synthetic multi-epoch parameter-space map combining `a_star`, inclination, luminosity, `f_col` evolution, weak inner stress, and posterior coverage. This remains provisional until full ADS review.

Did earlier work report full bias and frequentist/Bayesian coverage maps?

Provisional answer: not for the exact planned experiment. Existing works report model comparison, source-specific spin constraints, GRMHD-vs-NT biases, or propagated uncertainties. I did not verify a broad simulation-based calibration or posterior coverage map for the combined multi-epoch misspecification design.

Did earlier work include retrograde through near-extremal prograde spin?

Provisional answer: existing disk models usually permit broad spin ranges, and some tools allow retrograde spin, but the audited uncertainty studies do not appear to publish a complete retrograde-to-near-extremal prograde bias and coverage map under the combined `f_col` plus stress misspecification. This must be checked in full papers, not only abstracts.

Did earlier work publish reusable open Python code and all configurations?

Provisional answer: no exact match found. Several strong tools exist, but many are XSPEC local models, compiled ray tracers, transfer-table models, or source-specific packages. grtrans and NKBB/RayTransfer repositories exist, and Mummery et al. state their continuum-fitting model is publicly available. I did not find an open Python package with all configurations for the proposed simulation-based UQ campaign.

What exact new scientific statement can this project support?

Only after validation and confirmatory runs, the project could support a statement of this form:

> In a validated Kerr thin-disk synthetic experiment with declared luminosity range and transfer accuracy, joint multi-epoch fitting reduces random spin uncertainty under same-model assumptions, but its ability to reduce systematic spin bias depends on whether spectral hardening and inner-edge stress are correctly modeled; the project maps the parameter regimes where spin recovery is biased, calibrated, or non-identifiable.

This must not be stated as a result until the campaign is run.

## Go/No-Go Recommendation

Recommendation: conditional GO for Phase 1 only after human review of this Phase 0 audit.

The primary question still appears defensible, but it should be narrowed to avoid colliding with recent plunging-region/fullkerr observational work. The safest framing is not "discovering" stress or plunging emission. It is a controlled uncertainty-quantification experiment:

- synthetic, detector-independent first;
- zero-torque model misspecification tested against controlled `f_col` and weak inner-stress departures;
- explicit multi-epoch versus single-epoch comparison;
- posterior coverage and identifiability maps;
- complete spin interval if runtime permits.

## Required Refinements Before Coding Physics

1. Read the full text of Mummery et al. 2024, Mummery et al. 2025, Salvesen and Miller 2021, Kulkarni et al. 2011, and Agol and Krolik 2000 before finalizing the science grid.
2. Verify whether the Mummery/fullkerr model already provides public code and whether its parameterization should be treated as the primary inner-stress competitor.
3. Decide whether the inner-edge extension is "weak ISCO stress outside the ISCO" following Agol-Krolik or "plunging-region emission inside the ISCO" following recent fullkerr work. These are related but not identical.
4. Keep `f_col(L/L_Edd)` laws as sensitivity models unless parameters are taken from atmosphere calculations such as BHSPEC or other cited atmosphere grids.
5. Predeclare coverage metrics before any large simulation run.
6. Keep source-specific observational claims out of the first paper unless instrument folding and real-data calibration are added as a later phase.

## Fallback Questions

Fallback A: Multi-epoch identifiability of the `a_star`-`f_col`-inclination degeneracy under a zero-torque Novikov-Thorne simulator, with no inner-stress extension in the first paper.

Fallback B: Retrograde-to-prograde systematic map for color-correction misspecification only, using a validated transfer map and same-model coverage calibration.

Fallback C: Certified spectral emulator for Kerr thin-disk spectra, if recent literature already covers the science question but lacks a reproducible Python emulator with declared interpolation errors.

## Stop Conditions Carried Forward

- Stop if full-text review shows the exact combined multi-epoch misspecification and coverage map has already been published.
- Stop if the selected inner-stress prescription cannot be tied to a primary source and a well-defined luminosity increment.
- Stop if the Page-Thorne flux convention cannot be reconciled with the luminosity-efficiency test.
- Stop if same-model injection recovery or coverage fails before misspecification science begins.
