# Phase 12 Analysis Freeze

Status: frozen before hidden truths are joined to final summaries.

Config version: `phase12_ray_traced_transfer_v5_joint`.
Locked conditions: `data\processed\screening\phase11_confirmatory_conditions.csv`.
Frozen protocol: `data\processed\screening\phase11_confirmatory_protocol.md`.
Replicates per condition: 100.
Model backend: `ray_traced_transfer`.
Base spin grid count: 81.
Higher-resolution spin grid count: 121.
Energy bin count: 24.
Bias stability tolerance: 0.01.
Ray screen size: 24x24.
Ray screen alpha range: [-112.0, 112.0].
Ray screen beta range: [-112.0, 112.0].
Ray observer radius: 1000.0.
Ray disk outer radius: 80.0.
Ray step size: 0.1.
Ray max steps: 6000.
Limb darkening: `electron_scattering`.

Blinded replicate fits were written before unblinded condition
summaries. The hidden-truth CSV is separate and should not be read
by analysis scripts before this freeze point.
