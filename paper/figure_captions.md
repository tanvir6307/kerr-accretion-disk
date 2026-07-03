# Figure Captions

## Figure 1: `figure1_model_schematic.png`

Model schematic for the corrected v5 workflow. Boxes show the analysis sequence from locked conditions through the physically normalized, full-disk ray-traced spectra to the joint marginalized spin fit and the bias, coverage, and convergence summaries.

## Figure 2: `figure2_validation_residuals.png`

Independent validation margins from data/processed/validation_summary.csv. Each bar is the number of orders of magnitude the residual sits below its declared tolerance (capped at 15 for residuals at machine precision); all checks pass. Varied quantities include ISCO radius, efficiency, ray invariants, Page-Thorne flux, and a constant-intensity transfer-spectrum check.

## Figure 3: `figure3_disk_flux_profiles.png`

Disk flux profiles for a*=(-0.5, 0, 0.5, 0.9), normalized to dimensionless one-face flux with mass_accretion_rate=0.1. Solid lines are zero-torque Page-Thorne profiles; dashed lines use Delta_eta=0.02.

## Figure 4: `figure4_transfer_images.png`

Ray-traced 64x64 full-disk transfer-map image-plane samples for two benchmark spin/inclination pairs. Color is the transfer-map redshift g for disk-hit rays; alpha and beta are local observer-screen coordinates in gravitational radii.

## Figure 5: `figure5_ray_traced_spectra.png`

Ray-traced spectra for varied spin and stress at inclination 40 deg, eddington_ratio=0.1, f_col=1.7, for the fiducial 10 solar-mass black hole at 8 kpc. Uses the corrected v5 full-disk 64x64 transfer map with physical absolute normalization; bin energy flux is in erg/s/cm^2.

## Figure 6: `figure6_bias_width_proxy.png`

Condition-level absolute spin bias versus mean 68% interval width from the v5 joint marginalized-fit confirmatory table (color correction marginalized). Color is log10(mean chi2_per_dof); conditions with elevated chi2 are the inner-stress-misspecified cases that remain biased with narrow intervals.

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
