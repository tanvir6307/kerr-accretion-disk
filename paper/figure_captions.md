# Figure Captions

## Figure 1: `figure1_model_schematic.png`

Model schematic for the Phase 12 v4 workflow. Boxes show the fixed analysis sequence from locked conditions to bias, coverage, and convergence summaries; no numerical normalization is implied.

## Figure 2: `figure2_validation_residuals.png`

Independent validation residuals from data/processed/validation_summary.csv. Each bar is normalized by its declared tolerance; varied quantities include ISCO radius, efficiency, ray invariants, Page-Thorne flux, and a constant-intensity transfer-spectrum check.

## Figure 3: `figure3_disk_flux_profiles.png`

Disk flux profiles for a*=(-0.5, 0, 0.5, 0.9), normalized to dimensionless one-face flux with mass_accretion_rate=0.1. Solid lines are zero-torque Page-Thorne profiles; dashed lines use Delta_eta=0.02.

## Figure 4: `figure4_transfer_images.png`

Ray-traced 5x5 transfer-map image-plane samples for two benchmark spin/inclination pairs. Color is the transfer-map redshift g for disk-hit rays; alpha and beta are local observer-screen coordinates.

## Figure 5: `figure5_ray_traced_spectra.png`

Ray-traced detector-independent spectra for varied spin and stress at inclination 40 deg, eddington_ratio=0.1, f_col=1.7, and the Phase 12 v4 5x5 transfer-map normalization. Flux normalization is arbitrary and consistent across curves.

## Figure 6: `figure6_bias_width_proxy.png`

Condition-level bias versus mean 68% interval width from the Phase 12 v4 confirmatory table. Color is log10(mean chi2_per_dof). This is an identifiability proxy, not a multi-parameter posterior degeneracy plot.

## Figure 7: `figure7_multi_epoch_status.png`

Two-epoch shared-spin comparison from phase13p5_multi_epoch_summary.csv. Left panel compares the mean 68% spin-interval width from separate single-epoch fits against a joint two-epoch fit. Right panel shows the change in absolute spin bias; negative values favor the joint fit.

## Figure 8: `figure8_bias_map.png`

Mean spin-bias map from phase12_results_unblinded.csv. Cells aggregate over luminosity, f_col_true, and stress at fixed true spin and inclination; bias is posterior-mean spin minus injected spin.

## Figure 9: `figure9_coverage_map.png`

Mean 68% interval coverage by true spin and inclination from phase12_results_unblinded.csv. Cells aggregate over luminosity, f_col_true, and stress.

## Figure 10: `figure10_failure_domain.png`

Failure-domain summary from phase12_results_unblinded.csv. Left panel shows fit quality; right panel shows recorded failure rate. Color encodes absolute mean spin bias.

## Figure 11: `figure11_numerical_convergence.png`

Numerical convergence summary. Left panel uses Phase 12 v4 base versus higher-resolution spin-grid bias differences with the 0.01 tolerance. Right panel uses Phase 12.5 transfer spectra normalized by relative L1 difference to the 7x7 screen reference.
