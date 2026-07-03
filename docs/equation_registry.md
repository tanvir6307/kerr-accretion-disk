# Equation Registry

This registry records every production physical equation implemented in the
package. Phase 2 uses geometric units with `G = M = c = 1` and
Boyer-Lindquist coordinates ordered as `(t, r, theta, phi)`.

## Kerr Metric

- Source: Bardeen, Press, and Teukolsky (1972), The Astrophysical Journal
  178, 347, equations (2.1)-(2.3).
- Coordinates: Boyer-Lindquist `(t, r, theta, phi)`.
- Symbols:
  - `a_star`: dimensionless Kerr spin, equal to the usual geometric Kerr
    parameter `a/M` when `M=1`.
  - `Sigma = r^2 + a_star^2 cos(theta)^2`.
  - `Delta = r^2 - 2 r + a_star^2`.
  - `A = (r^2 + a_star^2)^2 - a_star^2 Delta sin(theta)^2`.
- Unit convention: dimensionless geometric units, `G = M = c = 1`.
  Dimensional radii are restored by multiplying by `r_g = GM/c^2`.
- Implemented in:
  - `src/kerrdisk/metric.py::kerr_auxiliary`
  - `src/kerrdisk/metric.py::covariant_metric`
  - `src/kerrdisk/metric.py::contravariant_metric`
- Expected limiting behavior:
  - `a_star = 0` gives Schwarzschild in curvature coordinates.
  - `g_{mu alpha} g^{alpha nu}` equals the identity outside coordinate
    singularities.
- Tests:
  - explicit Schwarzschild-component checks;
  - inverse residual property tests outside the horizon;
  - finite-value checks over the declared spin domain.

## Kerr Horizon

- Source: Bardeen, Press, and Teukolsky (1972), equation (2.3) with the
  event horizon defined by the outer root of `Delta = 0`; figure caption
  identifies `r_+`.
- Equation: `r_+ = 1 + sqrt(1 - a_star^2)`.
- Coordinates and units: Boyer-Lindquist radius in `GM/c^2`.
- Implemented in `src/kerrdisk/metric.py::horizon_radius`.
- Expected limiting behavior:
  - `a_star = 0` gives `r_+ = 2`.
  - `|a_star| -> 1` gives `r_+ -> 1`.
- Tests:
  - exact Schwarzschild value;
  - spin-sign symmetry;
  - domain errors for `|a_star| >= 1`.

## Equatorial ISCO Radius

- Source: Bardeen, Press, and Teukolsky (1972), equation (2.21).
- Coordinate and sign convention:
  - Boyer-Lindquist equatorial circular orbits.
  - Public API uses signed `a_star`: positive means prograde disk relative
    to the black-hole spin and negative means retrograde.
  - This implementation uses the guide convention
    `r_isco = 3 + Z2 - sign(a_star) sqrt((3 - Z1)(3 + Z1 + 2 Z2))`.
- Symbols:
  - `Z1 = 1 + (1 - a_star^2)^(1/3) [(1 + a_star)^(1/3) + (1 - a_star)^(1/3)]`.
  - `Z2 = sqrt(3 a_star^2 + Z1^2)`.
- Units: radius in `GM/c^2`.
- Implemented in `src/kerrdisk/isco.py::isco_radius`.
- Expected limiting behavior:
  - `a_star = 0` gives `r_isco = 6`.
  - `a_star -> 1` approaches `1` for prograde disks.
  - `a_star -> -1` approaches `9` for retrograde disks.
- Tests:
  - mandatory guide benchmark table at tolerance `1e-12`;
  - spin-domain validation;
  - monotonic decrease with increasing signed `a_star`.

## Circular Equatorial Orbits

- Source: Bardeen, Press, and Teukolsky (1972), equations (2.12)-(2.13)
  for specific energy and angular momentum, with angular velocity from the
  same circular-orbit section, equations (2.15)-(2.16). The implementation
  follows the guide's signed-`a_star` convention.
- Coordinates: equatorial Boyer-Lindquist circular orbits, `theta = pi/2`.
- Equations:
  - `Omega = 1 / (r^(3/2) + a_star)`.
  - `E = (r^(3/2) - 2 r^(1/2) + a_star)
     / (r^(3/4) sqrt(r^(3/2) - 3 r^(1/2) + 2 a_star))`.
  - `L = (r^2 - 2 a_star r^(1/2) + a_star^2)
     / (r^(3/4) sqrt(r^(3/2) - 3 r^(1/2) + 2 a_star))`.
- Units:
  - `Omega` is dimensionless in `1/t_g`.
  - `E` is specific energy per unit rest mass.
  - `L` is dimensionless specific angular momentum in `GM/c`.
- Implemented in:
  - `src/kerrdisk/circular_orbits.py::angular_velocity`
  - `src/kerrdisk/circular_orbits.py::specific_energy`
  - `src/kerrdisk/circular_orbits.py::specific_angular_momentum`
  - `src/kerrdisk/circular_orbits.py::nt_efficiency`
- Expected limiting behavior:
  - Far from the hole, `Omega ~ r^(-3/2)`, `E -> 1`, and `L ~ sqrt(r)`.
  - At the ISCO, `1 - E` matches the guide's zero-torque,
    no-photon-capture efficiency table.
- Tests:
  - mandatory efficiency table;
  - finite values for radii outside ISCO;
  - explicit domain errors for invalid spin or non-circular-orbit radius.

## Metric Derivatives

- Source equation: derivatives are exact calculus on the registered Kerr
  covariant and contravariant metric components above, not independently
  introduced physical formulae. Only the radial and polar derivatives are
  nonzero because the metric is stationary and axisymmetric.
- Implementations:
  - `src/kerrdisk/metric.py::metric_derivatives` — centered finite differences
    of the covariant metric, retained as an independent oracle.
  - `src/kerrdisk/metric.py::covariant_metric_derivatives` — closed-form
    analytic derivatives of `g_mu_nu`.
  - `src/kerrdisk/metric.py::contravariant_metric_derivatives` — closed-form
    analytic derivatives of `g^mu_nu`, used in the geodesic force term.
- The analytic contravariant derivatives replace the earlier finite-difference
  stencil inside `geodesics.py::geodesic_rhs`. This removes four metric
  inversions per force evaluation and eliminates the finite-difference step as
  a numerical parameter.
- Tests:
  - returned derivative tensors have shape `(4, 4, 4)`;
  - `t` and `phi` derivatives are zero (stationarity and axisymmetry);
  - analytic covariant derivatives match the finite-difference oracle;
  - analytic contravariant derivatives match centered differences of
    `contravariant_metric`;
  - analytic derivatives satisfy the exact identity
    `d(g^-1) = -g^-1 (dg) g^-1` across the valid domain (Hypothesis);
  - domain errors outside the horizon and at the coordinate axis.

## Page-Thorne Zero-Torque Disk Flux

- Source: Page and Thorne (1974), The Astrophysical Journal 191, 499,
  equations (10b), (11b), (12), and (34). The source defines `F(r)` as the
  time-averaged radiant-energy flux through one disk face, measured in the
  orbiting matter frame. The zero-torque boundary condition is imposed at
  `r_ms`, the marginally stable orbit.
- Coordinates and metric-area convention:
  - Page and Thorne use near-equatorial coordinates `(t, r, z, phi)` with
    metric determinant factor `(-g)^(1/2) = exp(v + psi + mu)`.
  - For the Kerr specialization in their equation (15), `exp(v + psi + mu) = r`
    when `G = M = c = 1`.
  - This is not the full four-dimensional Boyer-Lindquist determinant
    `Sigma sin(theta)`. The Phase 3 implementation uses the Page-Thorne disk
    area convention `sqrt_minus_g_disk = r`.
- General conservation-law form implemented:

  ```text
  F(r) = mdot / (4 pi sqrt_minus_g_disk)
         * [-dOmega/dr] / (E - Omega L)^2
         * integral_{r_in}^{r} (E - Omega L) dL/dr' dr'
  ```

- Symbols:
  - `r_in = r_isco(a_star)` for the zero-torque baseline.
  - `mdot` is the rest-mass accretion rate in geometric units. The dimensionless
    implementation is linear in `mdot` and defaults to `mdot = 1`.
  - `E`, `L`, and `Omega` are the registered circular-orbit quantities from
    Phase 2.
- One-face or two-face flux:
  - `F(r)` is one-face local emitted flux.
  - Ring luminosity from both faces in the Page-Thorne conservation equations
    uses `4 pi sqrt_minus_g_disk F dr`.
  - The luminosity-at-infinity integrand includes the circular-orbit energy
    factor `E`: `dL_inf = 4 pi sqrt_minus_g_disk F E dr`.
- Unit convention:
  - Dimensionless geometry uses `G = M = c = 1`.
  - Dimensional radius is `R = r GM/c^2`.
  - A dimensionless one-face flux profile computed with `mdot = 1` converts to
    SI flux by multiplying by `c^6 Mdot / (G^2 M^2)`, where `Mdot` is in kg/s
    and `M` is in kg.
  - CGS flux in `erg s^-1 cm^-2` is `1000` times SI flux in `W m^-2`.
- Implemented in:
  - `src/kerrdisk/disk_flux.py::page_thorne_flux_profile`
  - `src/kerrdisk/disk_flux.py::luminosity_at_infinity`
  - `src/kerrdisk/disk_flux.py::local_radiated_luminosity`
  - `src/kerrdisk/disk_flux.py::flux_si_from_dimensionless`
  - `src/kerrdisk/disk_flux.py::flux_cgs_from_dimensionless`
- Expected limiting behavior:
  - `F(r_isco) = 0`.
  - `F(r) > 0` for `r > r_isco`.
  - At large radius, `F(r) -> 3 mdot / (8 pi r^3)`.
  - `integral 4 pi r F E dr -> mdot [1 - E(r_isco)]` as the outer radius
    approaches infinity.
- Tests:
  - ISCO boundary and positivity;
  - Newtonian large-radius normalization;
  - luminosity-at-infinity normalization against `nt_efficiency`;
  - radial-resolution convergence;
  - SI/CGS unit-conversion scaling.

## Physical Constants for Unit Conversion

- Source: NIST/CODATA values as exposed in SI definitions and CODATA 2018:
  - `c = 299792458 m/s` exactly.
  - `G = 6.67430e-11 m^3 kg^-1 s^-2`.
  - `h = 6.62607015e-34 J s` exactly.
  - `k_B = 1.380649e-23 J K^-1` exactly.
  - `sigma_SB = 5.670374419e-8 W m^-2 K^-4` exactly.
- Implemented in `src/kerrdisk/constants.py`.

## Agol-Krolik Inner-Edge Stress Flux

- Source: Agol and Krolik (2000), "Magnetic Stress at the Marginally Stable
  Orbit: Altered Disk Structure, Radiation, and Black Hole Spin Evolution",
  equations (3), (6), and (8). Their equation (6) defines the inner-boundary
  condition in terms of an added radiative efficiency `Delta epsilon`, and the
  text below equation (6) states that this choice makes the integrated
  additional dissipation match `Delta epsilon`.
- Coordinates and assumptions:
  - Boyer-Lindquist radius `r` in `GM/c^2`.
  - The disk inner edge is fixed at `r_ms = r_isco(a_star)`.
  - The implementation adds the Agol-Krolik stress term outside the ISCO only;
    it does not add plunging-region emission, returning radiation, photon
    capture, magnetic evolution, or a GRMHD disk solution.
- Auxiliary function from Agol and Krolik equation (3):

  ```text
  C(r) = 1 - 3/r + 2 a_star / r^(3/2)
  C_ms = C(r_ms)
  ```

- Added one-face surface flux from the first term in Agol and Krolik
  equation (8), in the same dimensionless `G = M = c = 1` convention used for
  the Page-Thorne flux:

  ```text
  F_AK_extra(r) = 3 mdot / (8 pi r^3)
                  * r_ms^(3/2) C_ms^(1/2) Delta_eta
                    / [C(r) r^(1/2)]
  F_total(r) = F_Page-Thorne(r) + F_AK_extra(r)
  ```

- Parameter convention:
  - `Delta_eta` is the additional radiative efficiency relative to the
    zero-torque Page-Thorne / Novikov-Thorne baseline.
  - Production code restricts this controlled weak-stress sensitivity model to
    `0 <= Delta_eta <= 0.1`. Values outside this range require a new documented
    science decision before use.
  - `Delta_eta = 0` exactly recovers the Page-Thorne baseline.
- One-face or two-face flux:
  - `F_AK_extra(r)` is one-face local emitted flux.
  - The integrated added luminosity at infinity uses the same convention as the
    Page-Thorne validation:

    ```text
    Delta L_inf = integral 4 pi r F_AK_extra(r) E(r) dr
                -> mdot Delta_eta
    ```

- Expected limiting behavior:
  - `F_AK_extra(r) >= 0` for `r >= r_ms` in the adopted spin and
    `Delta_eta` domain.
  - At large radius, `C(r) -> 1`, so `F_AK_extra(r) proportional to r^(-7/2)`,
    matching the asymptotic behavior stated below Agol and Krolik equation (8).
- Implemented in:
  - `src/kerrdisk/disk_flux.py::agol_krolik_stress_flux`
  - `src/kerrdisk/disk_flux.py::stressed_page_thorne_flux_profile`
  - `src/kerrdisk/disk_flux.py::stress_luminosity_at_infinity`
- Tests:
  - zero-stress recovery of the Page-Thorne baseline;
  - nonnegative added flux in the adopted domain;
  - integrated added luminosity at infinity matches `mdot Delta_eta`;
  - large-radius `r^(-7/2)` slope;
  - invalid `Delta_eta` values are rejected.

## Effective Temperature

- Source: Stefan-Boltzmann law, with constant value from NIST/CODATA.
- Equation: `T_eff = (F / sigma_SB)^(1/4)`.
- Unit convention:
  - `F` is one-face local emitted flux in SI units `W m^-2`.
  - `T_eff` is in kelvin.
- Implemented in `src/kerrdisk/atmosphere.py::effective_temperature_from_flux`.
- Expected limiting behavior:
  - `F = 0` gives `T_eff = 0`.
  - Positive flux maps monotonically to positive temperature.
- Tests:
  - round-trip `sigma_SB T^4`;
  - invalid negative flux rejection.

## Planck Spectrum and Local Intensity

- Source: Planck radiation law for spectral radiance per unit frequency.
- Equation:

  ```text
  B_nu(T) = 2 h nu^3 / c^2 / [exp(h nu / (k_B T)) - 1]
  ```

- Unit convention:
  - `nu` is frequency in Hz.
  - `T` is temperature in kelvin.
  - `B_nu` is spectral radiance / specific intensity in
    `W m^-2 sr^-1 Hz^-1`.
- Adopted disk-emission convention:
  - baseline emission is isotropic in the comoving frame;
  - no limb darkening, returning radiation, absorption, or Compton tail;
  - bolometric one-face flux is `pi * integral B_nu dnu = sigma_SB T^4`.
- Implemented in:
  - `src/kerrdisk/atmosphere.py::planck_nu`
  - `src/kerrdisk/atmosphere.py::integrate_specific_intensity`
  - `src/kerrdisk/atmosphere.py::bolometric_intensity`
  - `src/kerrdisk/atmosphere.py::isotropic_angular_factor`
- Numerical convention:
  - the exponential is evaluated with `expm1` for small and moderate `x`;
  - bins with `h nu / (k_B T) > 700` are assigned zero intensity to avoid
    floating-point overflow in the Wien tail.
- Tests:
  - numerical bolometric integral agrees with `sigma_SB T^4 / pi`;
  - no overflow or underflow warnings for broad frequency/temperature ranges;
  - angular factor is unity for the isotropic baseline.

## Diluted Blackbody Color Correction

- Source: Shimura and Takahara (1995), ApJ 445, 780, for the accretion-disk
  spectral-hardening / diluted-blackbody context. Phase 4 implements the guide's
  controlled local sensitivity model, not a full atmosphere calculation.
- Equation:

  ```text
  I_nu,em = f_col^-4 B_nu(f_col T_eff)
  ```

- Unit convention:
  - `T_eff` is kelvin.
  - `f_col` is dimensionless and positive.
  - `I_nu,em` is in `W m^-2 sr^-1 Hz^-1`.
- Implemented in:
  - `src/kerrdisk/atmosphere.py::diluted_blackbody_nu`
  - `src/kerrdisk/atmosphere.py::ConstantFcol`
  - `src/kerrdisk/atmosphere.py::EpochwiseFcol`
  - `src/kerrdisk/atmosphere.py::LuminosityLawFcol`
- Clipping convention:
  - `LuminosityLawFcol` follows the guide's controlled
    `clip[f0 + f1 log10(ell / pivot), lower, upper]` law.
  - Clipping is exposed through `FcolEvaluation.clipped`; callers must record
    this in run metadata for production simulations.
- Tests:
  - bolometric flux is preserved under `f_col`;
  - constant, epochwise, and luminosity-law models validate inputs;
  - luminosity-law clipping is visible to callers.

## Hamiltonian Null Geodesics

- Sources:
  - Hamiltonian form follows the standard canonical geodesic equations for a
    metric Hamiltonian and the Phase 5 project guide.
  - Carter (1968), Physical Review 174, 1559, for separability and the Carter
    constant.
  - Bardeen, Press, and Teukolsky (1972), for Kerr constants of motion and
    black-hole shadow context.
- Coordinates: Boyer-Lindquist `(t, r, theta, phi)`.
- Hamiltonian:

  ```text
  H = 1/2 g^{mu nu} p_mu p_nu = 0
  dx^mu/dlambda = g^{mu nu} p_nu
  dp_mu/dlambda = -1/2 partial_mu(g^{alpha beta}) p_alpha p_beta
  ```

- Numerical convention:
  - `p_t` and `p_phi` derivatives are zero because the metric is stationary
    and axisymmetric.
  - `partial_r g^{alpha beta}` and `partial_theta g^{alpha beta}` are centered
    finite differences of the registered contravariant metric.
  - The scalar reference integrator is explicit RK4 with event checks between
    accepted steps. It is a validation baseline, not the final optimized ray
    engine.
- Invariants:
  - photon energy `E_gamma = -p_t`;
  - axial angular momentum `L_z = p_phi`;
  - Carter constant for null rays:

    ```text
    Q = p_theta^2 + cos(theta)^2 [L_z^2 / sin(theta)^2 - a_star^2 E_gamma^2]
    ```

- Implemented in:
  - `src/kerrdisk/geodesics.py::hamiltonian`
  - `src/kerrdisk/geodesics.py::geodesic_rhs`
  - `src/kerrdisk/geodesics.py::carter_constant_null`
  - `src/kerrdisk/geodesics.py::rk4_step`
- Tests:
  - null Hamiltonian is near zero for initialized rays;
  - `E_gamma`, `L_z`, and `Q` drift remains within declared scalar-reference
    tolerances for simple rays.

## Observer Tetrad and Screen Initialization

- Source: local orthonormal tetrad construction from the registered Kerr metric.
- Coordinates and observer:
  - static observer at large Boyer-Lindquist radius with fixed
    `(r_obs, theta_obs, phi_obs)`;
  - valid only where `g_tt < 0`, outside the ergosphere.
- Tetrad:
  - `e_(t)^mu = delta_t^mu / sqrt(-g_tt)`;
  - `e_(r)^mu = delta_r^mu / sqrt(g_rr)`;
  - `e_(theta)^mu = delta_theta^mu / sqrt(g_theta theta)`;
  - `e_(phi)^mu = A delta_t^mu + B delta_phi^mu`, with
    `A = -g_tphi B / g_tt` and
    `B = 1 / sqrt(g_phiphi - g_tphi^2 / g_tt)`.
- Screen direction convention:
  - local photon energy is set to unity;
  - `alpha` and `beta` are local screen-length coordinates at the observer;
  - the local spatial direction is proportional to
    `-e_(r) + (beta/r_obs) e_(theta) + (alpha/r_obs) e_(phi)`.
  - This is a documented large-radius camera convention for the scalar
    reference tracer and must be independently cross-validated before
    production transfer maps.
- Implemented in:
  - `src/kerrdisk/raytrace.py::static_observer_tetrad`
  - `src/kerrdisk/raytrace.py::initial_photon_covector`
- Tests:
  - tetrad orthonormality;
  - initial photon null constraint;
  - Schwarzschild shadow transition is bracketed around `sqrt(27)`.

## Ray Outcomes

- Outcomes are exactly:
  - `DISK_HIT`
  - `HORIZON_CAPTURE`
  - `ESCAPED`
  - `MAX_STEPS`
  - `NUMERICAL_FAILURE`
- Implemented in:
  - `src/kerrdisk/raytrace.py::RayOutcome`
  - `src/kerrdisk/raytrace.py::trace_ray`
- Event conventions:
  - horizon capture when `r <= r_+ (1 + epsilon_h)`;
  - escape when `r >= r_escape` after outward motion;
  - disk hit when `theta - pi/2` changes sign and the interpolated radius lies
    inside `[r_inner, r_outer]`;
  - `MAX_STEPS` is explicit and never silently treated as success.

## Emitter Four-Velocity and Redshift

- Sources:
  - Bardeen, Press, and Teukolsky (1972), circular-orbit angular velocity from
    the same orbit equations registered in Phase 2.
  - Cunningham (1975), ApJ 202, 788, for relativistic disk transfer/redshift
    treatment.
- Coordinates: Boyer-Lindquist equatorial disk, `theta = pi/2`.
- Circular emitter four-velocity:

  ```text
  u^mu = u^t (1, 0, 0, Omega)
  u^t = [-g_tt - 2 Omega g_tphi - Omega^2 g_phiphi]^(-1/2)
  ```

- Frequency measured by an observer with four-velocity `u^mu`:

  ```text
  nu = -p_mu u^mu
  ```

- Redshift factor:

  ```text
  g = nu_obs / nu_em
    = (-p_mu u_obs^mu) / (-p_mu u_em^mu)
  ```

- Implemented in:
  - `src/kerrdisk/spectrum.py::circular_emitter_four_velocity`
  - `src/kerrdisk/spectrum.py::measured_frequency`
  - `src/kerrdisk/spectrum.py::redshift_factor`
  - `src/kerrdisk/spectrum.py::emission_angle_cosine`
- Tests:
  - emitter normalization `u_mu u^mu = -1`;
  - static-to-static Schwarzschild redshift sanity check;
  - normal-emission Schwarzschild photon gives `emission_mu = 1`;
  - positive redshift factors for disk-hit transfer-map records.

## Transfer Map and Observed Spectrum

- Sources:
  - Lindquist (1966), Annals of Physics 37, 487, for general-relativistic
    transport context.
  - Cunningham (1975), for Kerr disk transfer maps using redshift and emission
    angle.
- Invariant intensity relation:

  ```text
  I_nu / nu^3 = invariant
  I_nu,obs(nu_obs) = g^3 I_nu,em(nu_obs / g)
  ```

- Observer-screen solid angle:

  ```text
  dOmega_obs = dAlpha dBeta / D^2
  ```

  where `D` is the observer distance in the same geometric length units as
  screen coordinates.
- Observed flux density:

  ```text
  F_nu,obs = sum_i g_i^3 I_nu,em(nu_obs/g_i, r_i) dOmega_i
  ```

- Emission-angle convention:
  - `emission_mu = |p_mu n^mu| / (-p_mu u_em^mu)`, where `n^mu` is the
    equatorial disk normal in the circular-emitter frame.
  - Values are clipped to `[0, 1]` only to absorb floating-point roundoff.
- Implemented in:
  - `src/kerrdisk/spectrum.py::build_transfer_map`
  - `src/kerrdisk/spectrum.py::observed_flux_density`
  - `src/kerrdisk/spectrum.py::save_transfer_map`
  - `src/kerrdisk/spectrum.py::load_transfer_map`
- Tests:
  - `D^-2` scaling through screen solid angle;
  - symmetry for a face-on Schwarzschild observer;
  - cache round trip preserves transfer-map values;
  - one transfer map is reused for multiple frequency grids.

## Synthetic Spectra and Likelihoods

- Source: standard statistical definitions for independent Gaussian
  measurements and independent Poisson counts. No detector response matrix is
  introduced in Phase 9.
- Energy-bin convention:
  - Energy bin edges are in keV and strictly increasing.
  - Bin centers are geometric means, `E_i = sqrt(E_lo,i E_hi,i)`, because the
    detector-independent development grids are logarithmic.
  - Bin widths are `Delta E_i = E_hi,i - E_lo,i`.
  - If a callable photon-flux density model `phi(E)` is supplied in
    `photons cm^-2 s^-1 keV^-1`, the bin-integrated photon flux is evaluated by
    the midpoint rule:

    ```text
    Phi_i = phi(E_i) Delta E_i
    ```

  - The stored bin energy flux is in `keV cm^-2 s^-1 bin^-1`:

    ```text
    S_i = E_i Phi_i
    ```

- Gaussian relative-error debug noise:

  ```text
  sigma_i = f_rel mu_i
  y_i = mu_i + sigma_i z_i,  z_i ~ Normal(0, 1)
  ln L = -1/2 sum_i [ (y_i - mu_i)^2 / sigma_i^2
                     + ln(2 pi sigma_i^2) ]
  ```

  where `mu_i` is the noiseless expectation stored for the epoch.
- Poisson count noise:

  ```text
  lambda_i = Phi_i A_eff t_exp
  k_i ~ Poisson(lambda_i)
  ln L = sum_i [ k_i ln(lambda_i) - lambda_i - ln(k_i!) ]
  ```

  where `A_eff` is a scalar effective area in `cm^2` and `t_exp` is exposure in
  seconds. Phase 9 intentionally uses no redistribution matrix, background, or
  calibration model.
- Randomness convention:
  - `condition_seed`, `epoch_seed`, and `noise_seed` are deterministic hashes of
    the master seed and labels.
  - Every generated epoch stores the derived seeds in a seed manifest.
- Implemented in:
  - `src/kerrdisk/synthetic.py::make_log_energy_bins`
  - `src/kerrdisk/synthetic.py::generate_gaussian_epoch`
  - `src/kerrdisk/synthetic.py::generate_poisson_epoch`
  - `src/kerrdisk/synthetic.py::generate_multi_epoch_dataset`
  - `src/kerrdisk/likelihood.py::gaussian_log_likelihood`
  - `src/kerrdisk/likelihood.py::poisson_log_likelihood`
  - `src/kerrdisk/likelihood.py::dataset_log_likelihood`
  - `src/kerrdisk/likelihood.py::maximize_scalar_likelihood`
- Tests:
  - deterministic seed derivation and replay;
  - Gaussian and Poisson noise moments against theory;
  - likelihood values match manual formulas;
  - invalid model expectations return explicit `-inf`;
  - high-signal same-model scalar injection recovery.

## Inference Diagnostics and Calibration

- Sources:
  - Gelman and Rubin (1992), "Inference from Iterative Simulation Using
    Multiple Sequences", Statistical Science 7, 457-472, for the multiple-chain
    potential scale reduction diagnostic.
  - Vehtari, Gelman, Simpson, Carpenter, and Buerkner (2021),
    "Rank-Normalization, Folding, and Localization: An Improved R-hat for
    Assessing Convergence of MCMC", Bayesian Analysis 16, 667-718, for modern
    R-hat and effective-sample-size workflow guidance.
  - Talts, Betancourt, Simpson, Vehtari, and Gelman (2018),
    "Validating Bayesian Inference Algorithms with Simulation-Based
    Calibration", arXiv:1804.06788, for posterior-rank calibration checks.
- Phase 10 scope:
  - The implemented sampler adapter is an internal random-walk Metropolis
    baseline for validation and API development.
  - It is not yet the final production sampler for science results.
- Uniform bounded prior for parameter `theta_j`:

  ```text
  p(theta_j) = 1 / (upper_j - lower_j),  lower_j <= theta_j <= upper_j
  log p(theta) = -sum_j log(upper_j - lower_j)
  ```

  Values outside the box return `-inf`. Unit-cube transform:

  ```text
  theta_j = lower_j + u_j (upper_j - lower_j),  0 <= u_j <= 1
  ```

- Posterior kernel:

  ```text
  log posterior(theta) = log likelihood(theta) + log prior(theta)
  ```

- Random-walk Metropolis proposal and acceptance:

  ```text
  theta' = theta + Normal(0, sigma_step)
  alpha = min[1, exp(log posterior(theta') - log posterior(theta))]
  ```

  Proposals outside the prior box are rejected through the `-inf` prior.
- Basic split-chain R-hat implemented for scalar diagnostics:

  ```text
  W = mean_s var_n(theta_s)
  B = n var_s(mean(theta_s))
  var_hat = ((n - 1) / n) W + B / n
  Rhat = sqrt(var_hat / W)
  ```

  where chains are split in half before computing chain means and variances.
- Effective sample size uses the initial-positive autocorrelation sequence:

  ```text
  ESS = N / [1 + 2 sum_t rho_t]
  ```

  with summation stopped at the first nonpositive autocorrelation term.
- Equal-tailed credible interval:

  ```text
  [quantile(samples, (1 - level)/2), quantile(samples, (1 + level)/2)]
  ```

- Posterior rank for SBC:

  ```text
  rank = count(samples < theta_true)
  ```

  A calibrated one-dimensional posterior sampler should produce approximately
  uniform ranks over repeated same-model simulations.
- Coverage estimate for level `q`:

  ```text
  coverage_q = mean[theta_true in CI_q(samples)]
  ```

- Implemented in:
  - `src/kerrdisk/inference.py::UniformPrior`
  - `src/kerrdisk/inference.py::PosteriorProblem`
  - `src/kerrdisk/inference.py::optimize_posterior`
  - `src/kerrdisk/inference.py::run_random_walk_metropolis`
  - `src/kerrdisk/inference.py::split_rhat`
  - `src/kerrdisk/inference.py::effective_sample_size`
  - `src/kerrdisk/inference.py::simulation_based_calibration`
- Tests:
  - prior transform and log-prior rejection;
  - optimizer recovery for a Gaussian target;
  - repeated-chain random-walk sampler agreement on a Gaussian target;
  - R-hat and ESS convergence diagnostics;
  - prior predictive finite-output checks;
  - SBC rank and 68%/95% coverage summaries for a calibrated analytic
    same-model posterior.

## Phase 11 Screening Metrics

- Source: project-defined screening estimands from the statistical plan in
  `KERR_DISK_CODEX_RESEARCH_GUIDE.md`, Section 20. These are summary
  statistics, not new physical disk equations.
- Scope:
  - Phase 11 uses a lightweight detector-independent proxy spectrum to exercise
    the campaign machinery, failure accounting, and refinement selection.
  - The proxy formula is not a Kerr disk spectrum and must not be used for
    astrophysical conclusions.
- For condition `c` and replicate `r`, the scalar screening estimand is spin
  bias:

  ```text
  Delta a_*(c, r) = a_hat(c, r) - a_true(c)
  ```

- Condition-level summaries:

  ```text
  mean_bias = mean_r Delta a_*(c, r)
  median_bias = median_r Delta a_*(c, r)
  rmse = sqrt(mean_r [Delta a_*(c, r)]^2)
  width_q = upper_q - lower_q
  coverage_q = mean_r [a_true(c) in CI_q(c, r)]
  failure_rate = failed_replicates / planned_replicates
  ```

  where `q` is `0.68` or `0.95`, and `CI_q` is an equal-tailed posterior
  interval from the one-dimensional spin posterior grid.
- Fit-quality screening statistic:

  ```text
  chi2_per_dof = sum_i [(y_i - mu_i)^2 / sigma_i^2] / (N_bins - N_fit)
  ```

  with explicit rejection if `N_bins <= N_fit`.
- Refinement flags:
  - `high_bias_flag = abs(mean_bias) >= high_bias_threshold`;
  - `low_identifiability_flag = mean_width_68 >= low_identifiability_width`;
  - `poor_fit_flag = mean_chi2_per_dof >= chi2_per_dof_max`;
  - `needs_refinement` is true if any refinement flag is true or if
    `failure_rate > failure_rate_max`.
- Implemented in:
  - `src/kerrdisk/screening.py::build_screening_conditions`
  - `src/kerrdisk/screening.py::run_screening_campaign`
  - `src/kerrdisk/screening.py::write_screening_outputs`
- Tests:
  - deterministic condition construction;
  - every planned condition receives a status row;
  - high-bias and low-identifiability flags follow the declared thresholds;
  - no failed replicate is silently dropped from summaries;
  - machine-readable confirmatory condition lists and frozen protocol files are
    written before Phase 12.

## Phase 12 Confirmatory Campaign Metrics

- Source: project-defined confirmatory estimands from
  `KERR_DISK_CODEX_RESEARCH_GUIDE.md`, Section 20 and Phase 12. These are
  campaign summary statistics, not new physical disk equations.
- Scope:
  - Phase 12 consumes the locked Phase 11 condition list and frozen protocol.
  - Blinded condition IDs are deterministic hashes of the master seed and
    locked condition ID.
  - Hidden truths are written separately from blinded replicate fits before the
    unblinded summary table is generated.
- Replicate-level fitted quantities are the posterior mean spin, MAP spin,
  equal-tailed 68% and 95% spin intervals, and fit quality:

  ```text
  chi2_per_dof = sum_i [(y_i - mu_i)^2 / sigma_i^2] / (N_bins - 1)
  ```

- Condition-level unblinded summaries:

  ```text
  Delta a_*(r) = a_mean(r) - a_true
  mean_bias = mean_r Delta a_*(r)
  median_bias = median_r Delta a_*(r)
  rmse = sqrt(mean_r [Delta a_*(r)]^2)
  coverage_q = mean_r [a_true in CI_q(r)]
  failure_rate = failed_replicates / planned_replicates
  ```

- Numerical-resolution rerun comparison:

  ```text
  abs_bias_difference = abs(mean_bias_base - mean_bias_high_resolution)
  stable = abs_bias_difference <= bias_stability_abs
  ```

- Implemented in:
  - `src/kerrdisk/confirmatory.py::load_locked_conditions`
  - `src/kerrdisk/confirmatory.py::run_confirmatory_campaign`
  - `src/kerrdisk/confirmatory.py::write_confirmatory_outputs`
- Tests:
  - stable blinded IDs;
  - all planned replicates accounted for;
  - hidden-truth and blinded files are separated;
  - base and higher-resolution bias summaries are compared;
  - invalid configuration and missing frozen protocol are explicit failures.

## Phase 12 Kerr Thin-Disk Spectral Backend

- Source:
  - The radial one-face disk flux is the registered Page-Thorne zero-torque
    flux plus the registered Agol-Krolik stress extension when
    `Delta_eta > 0`.
  - The local emission spectrum follows the registered diluted-blackbody
    atmosphere convention.
  - The ray-traced backend uses the registered transfer-map invariant
    `I_nu / nu^3` relation, redshift factors, and observer-screen solid angles.
- Scope:
  - The `kerr_thin_disk` backend is a deterministic inclination-projected
    development backend retained for comparison.
  - The corrected `ray_traced_transfer` backend used for Phase 12 v4 traces
    image-plane screen rays, keeps only disk-hit transfer records, and sums
    redshifted local emission over screen solid angle.
  - The v4 backend includes physical transfer-map emission angles and a
    normalized electron-scattering limb-darkening option. It does not fold a
    detector response, polarization, or iterative returning-radiation
    reprocessing into the confirmatory likelihood.
- Energy-bin convention:

  ```text
  S_i = integral_{E_lo,i}^{E_hi,i} F_E dE
      approx F_E(E_center,i) Delta E_i
  ```

- Local effective temperature proxy:

  ```text
  T_eff(r) = T0 [F(r)]^(1/4)
  ```

  where `F(r)` is the dimensionless registered one-face flux and `T0` is the
  configured keV scale for the development confirmatory backend.
- Diluted blackbody spectral shape in keV units:

  ```text
  I_E(E, r) proportional to f_col^(-4) E^3 /
      {exp[E / (f_col T_eff(r))] - 1}
  ```

- Disk integration:

  ```text
  F_E(E) proportional to cos(i) integral 2 pi r I_E(E, r) dr
  ```

  with a strictly increasing logarithmic radial grid from ISCO to the configured
  disk outer radius.
- Ray-traced transfer-map integration:

  ```text
  F_E,obs(E_obs) proportional to
      sum_j g_j^3 I_E,em(E_obs / g_j, r_j) DeltaOmega_j
  S_i approx F_E,obs(E_center,i) Delta E_i
  ```

  where each `j` is a successful image-plane disk-hit ray, `g_j` is the
  registered redshift factor, and `DeltaOmega_j` is the screen-cell solid
  angle. For `limb_darkening = electron_scattering`, the local intensity is
  multiplied by `0.5 + 0.75 emission_mu`; isotropic emission uses factor `1`.
- Implemented in:
  - `src/kerrdisk/thermal_spectrum.py::kerr_thin_disk_energy_flux`
  - `src/kerrdisk/thermal_spectrum.py::ray_traced_kerr_thin_disk_energy_flux`
  - `src/kerrdisk/confirmatory.py::run_confirmatory_campaign`
- Tests:
  - finite positive spectra;
  - spin changes spectral shape;
  - stress changes the spectrum relative to zero torque;
  - ray-traced transfer-map spectra are finite and positive;
  - limb darkening changes the ray-traced spectrum;
  - confirmatory campaigns can run with the `ray_traced_transfer` backend;
  - invalid inputs fail explicitly.

## Phase 12.5 Transfer Diagnostics

- Scope:
  - `src/kerrdisk/radiation_diagnostics.py` samples locally emitted photons
    from circular disk emitters and classifies terminal outcomes as escaped,
    horizon-captured, returning to the disk, max-steps, or numerical failure.
  - `src/kerrdisk/transfer_validation.py` generates screen-resolution
    convergence rows, capture/returning diagnostic rows, and an external
    transfer-map comparison row when an external ray-tracer CSV is supplied.
- Local emitted-photon convention:
  - the emitter frame uses `u^mu`, a radial unit vector, the equatorial disk
    normal, and an azimuthal unit vector orthogonal to `u^mu`;
  - local photon energy is unity;
  - `emission_mu` and `emission_azimuth` parameterize the upper disk
    hemisphere.
- Implemented in:
  - `src/kerrdisk/radiation_diagnostics.py::emitted_photon_state`
  - `src/kerrdisk/radiation_diagnostics.py::sample_emission_outcomes`
  - `src/kerrdisk/transfer_validation.py::run_transfer_convergence`
  - `src/kerrdisk/transfer_validation.py::run_external_transfer_comparison`
- Tests:
  - emitted photons satisfy the null Hamiltonian constraint;
  - sampled outcome accounting conserves the sample count;
  - invalid local emission inputs fail explicitly.

## Physical Scales and Observed Normalization

- Purpose: convert the dimensionless geometric-unit disk physics into an
  absolutely normalized observed spectrum for a physical system defined by
  black-hole mass, distance, inclination, and Eddington ratio. This replaces the
  earlier arbitrary `temperature_scale_kev` shape parameter.
- Scale relations (SI):
  - Gravitational radius `r_g = G M / c^2`.
  - Eddington luminosity (hydrogen, Thomson scattering)
    `L_Edd = 4 pi G M m_p c / sigma_T`.
  - Accretion rate from the Eddington ratio `ell = L / L_Edd` and radiative
    efficiency `eta`: `Mdot = ell L_Edd / (eta c^2)`. For the zero-torque disk
    `eta = 1 - E(r_isco)`; for the controlled stress model the total efficiency
    `eta = [1 - E(r_isco)] + Delta_eta` is used so `ell` remains `L_total/L_Edd`.
  - Photon frequency from energy `nu = E_keV * (10^3 e) / h`.
- Effective temperature: `T_eff(r) = [F_SI(r) / sigma_SB]^{1/4}`, with `F_SI`
  from the registered Page-Thorne SI conversion evaluated at the physical `Mdot`.
- Local emission: the registered diluted blackbody
  `I_nu,em = B_nu(f_col T_eff) / f_col^4`.
- Observed flux: `F_nu,obs = sum_hits g^3 D(mu) I_nu,em(nu_obs/g) dOmega`, where
  `g` is the redshift factor, `D(mu)` the limb-darkening factor, and `dOmega` the
  observer solid angle per image pixel. With screen coordinates and the
  astronomical distance both expressed in `r_g`, the transfer map's stored
  `solid_angle = d_alpha d_beta / observer_distance^2` is the physical solid
  angle in steradians, so the `r_g^2` pixel-area factor and the `D^-2` distance
  factor are captured automatically. The binned observable is
  `F_nu,obs * d_nu` per bin, reported in `erg s^-1 cm^-2`.
- Implemented in:
  - `src/kerrdisk/scales.py`
  - `src/kerrdisk/thermal_spectrum.py::_effective_temperature_profile`
  - `src/kerrdisk/thermal_spectrum.py::kerr_thin_disk_energy_flux`
  - `src/kerrdisk/thermal_spectrum.py::ray_traced_kerr_thin_disk_energy_flux`
- Expected behavior and tests:
  - `r_g`, `L_Edd`, and `Mdot` reproduce standard values for a ten-solar-mass
    black hole; the accretion rate reproduces the target luminosity;
  - the observed flux scales as `D^-2`;
  - the peak effective temperature lies in the soft X-ray band for the fiducial
    ten-solar-mass system.

## Future Equations

Phase 13 and Phase 13.5 add no new physical equations. Their figures, tables,
captions, claim audit, multi-epoch comparison, release manifest, and checksums
are generated from registered equations and final machine-readable outputs.
