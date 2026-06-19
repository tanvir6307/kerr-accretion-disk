# Phase 11 Frozen Confirmatory Protocol

Status: frozen before Phase 12 confirmatory truths are generated.

This protocol is based on Phase 11 screening artifacts only. It does
not state final astrophysical conclusions.

## Refinement Criteria

- High bias: `abs(mean_bias) >= 0.05`.
- Low identifiability: `mean_width_68 >= 0.25`.
- Poor fit: `mean_chi2_per_dof >= 2.0`.
- Failure accounting: `failure_rate > 0.0`.

Any condition meeting at least one criterion is included in
`phase11_confirmatory_conditions.csv`.

## Accounting Rules

- Failed replicates are retained in `phase11_replicates.csv`.
- Conditions are never removed post hoc without a status row and
  failure cause.
- Phase 12 must use this locked condition list unless a new protocol
  version is created before hidden truths are generated.

## Counts

- Screening conditions: 64.
- Conditions selected for refinement: 48.
- Planned replicates per condition: 20.
