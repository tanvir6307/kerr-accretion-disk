# Phase 13.5 Multi-Epoch Analysis Freeze

Status: frozen before Phase 14 manuscript drafting.

Config version: `phase13p5_joint_emulator_v5`.
Confirmatory config: `D:\Blackhole\configs\production\phase12_joint_v5.yaml`.
Model backend: `ray_traced_transfer`.
Groups: 24.
Epochs per group: 2.
Replicates per group: 30.
Spin grid count: 81.
Energy bin count: 24.
Ray screen size: 24x24.
Limb darkening: `electron_scattering`.

Each group pairs locked Phase 12 conditions with the same spin,
inclination, fitted color correction, true color correction, and
inner-stress setting, while treating the two luminosities as epochs.
The comparison fits each epoch separately and then jointly with a
shared spin parameter and fixed epoch-level luminosity metadata.
