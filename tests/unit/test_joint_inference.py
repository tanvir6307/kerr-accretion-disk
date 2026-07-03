"""Tests for multi-parameter joint spectral inference and model-based SBC."""

from collections.abc import Mapping

import numpy as np
import pytest

from kerrdisk.inference import ParameterSpec, UniformPrior
from kerrdisk.joint_inference import (
    initial_walker_ball,
    make_spectrum_posterior,
    named_forward_model,
    run_emcee_sampler,
    run_spectrum_sbc,
    summarize_joint_posterior,
)
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import KerrThinDiskSettings, kerr_thin_disk_energy_flux

_BINS = make_log_energy_bins(bin_count=6)
_SETTINGS = KerrThinDiskSettings(radial_grid_count=24, disk_outer_radius_rg=400.0)


def _base_model(args: Mapping[str, float]) -> np.ndarray:
    return kerr_thin_disk_energy_flux(
        a_star=args["spin"],
        inclination_deg=45.0,
        eddington_ratio=args.get("eddington_ratio", 0.1),
        f_col=args.get("f_col", 1.7),
        delta_eta=0.0,
        energy_bins=_BINS,
        settings=_SETTINGS,
    )


def _noisy_data(
    truth: Mapping[str, float],
    relative_error: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    expectation = np.asarray(_base_model(truth), dtype=np.float64)
    sigma = relative_error * expectation
    rng = np.random.default_rng(seed)
    return rng.normal(expectation, sigma), sigma * sigma


def test_joint_fit_recovers_injected_parameters() -> None:
    prior = UniformPrior(
        (
            ParameterSpec("spin", -0.9, 0.99),
            ParameterSpec("eddington_ratio", 0.03, 0.30),
            ParameterSpec("f_col", 1.4, 2.2),
        )
    )
    truth = {"spin": 0.7, "eddington_ratio": 0.12, "f_col": 1.7}
    data, variance = _noisy_data(truth, 0.02, seed=1)
    forward = named_forward_model(_base_model, ["spin", "eddington_ratio", "f_col"])
    problem = make_spectrum_posterior(prior, forward, data, variance)
    start = np.array([0.7, 0.12, 1.7])
    positions = initial_walker_ball(prior, start, walker_count=18, seed=0)

    chain = run_emcee_sampler(
        problem, initial_positions=positions, draws=180, burn_in=150, seed=5
    )
    summary = summarize_joint_posterior(prior, chain)

    spin = summary.by_name("spin")
    assert spin.ci68_lower < 0.7 < spin.ci68_upper
    assert spin.mean == pytest.approx(0.7, abs=0.05)


def test_free_color_correction_inflates_spin_uncertainty() -> None:
    truth = {"spin": 0.6, "eddington_ratio": 0.1, "f_col": 1.7}
    data, variance = _noisy_data(truth, 0.03, seed=2)

    fixed_prior = UniformPrior((ParameterSpec("spin", -0.9, 0.99),))
    fixed_forward = named_forward_model(
        _base_model, ["spin"], fixed={"eddington_ratio": 0.1, "f_col": 1.7}
    )
    fixed_problem = make_spectrum_posterior(fixed_prior, fixed_forward, data, variance)
    fixed_positions = initial_walker_ball(
        fixed_prior, np.array([0.6]), walker_count=12, seed=0
    )
    fixed_chain = run_emcee_sampler(
        fixed_problem, initial_positions=fixed_positions, draws=180, burn_in=150, seed=7
    )
    fixed_spin = summarize_joint_posterior(fixed_prior, fixed_chain).by_name("spin")

    free_prior = UniformPrior(
        (
            ParameterSpec("spin", -0.9, 0.99),
            ParameterSpec("f_col", 1.4, 2.2),
        )
    )
    free_forward = named_forward_model(
        _base_model, ["spin", "f_col"], fixed={"eddington_ratio": 0.1}
    )
    free_problem = make_spectrum_posterior(free_prior, free_forward, data, variance)
    free_positions = initial_walker_ball(
        free_prior, np.array([0.6, 1.7]), walker_count=12, seed=0
    )
    free_chain = run_emcee_sampler(
        free_problem, initial_positions=free_positions, draws=180, burn_in=150, seed=7
    )
    free_spin = summarize_joint_posterior(free_prior, free_chain).by_name("spin")

    fixed_width = fixed_spin.ci68_upper - fixed_spin.ci68_lower
    free_width = free_spin.ci68_upper - free_spin.ci68_lower

    # Marginalizing over the color correction inflates the spin uncertainty:
    # this is the spin-f_col degeneracy the study is designed to expose.
    assert free_width > 1.3 * fixed_width


def test_spectrum_sbc_is_approximately_calibrated() -> None:
    prior = UniformPrior(
        (
            ParameterSpec("spin", 0.0, 0.9),
            ParameterSpec("f_col", 1.5, 2.0),
        )
    )
    forward = named_forward_model(
        _base_model, ["spin", "f_col"], fixed={"eddington_ratio": 0.1}
    )

    result = run_spectrum_sbc(
        prior,
        forward,
        rank_parameter="spin",
        relative_error=0.1,
        simulations=12,
        walker_count=8,
        draws=120,
        burn_in=90,
        seed=11,
        rank_bins=4,
    )

    assert int(result.rank_histogram.sum()) == 12
    # Loose calibration bands: a small-simulation, fast SBC smoke check. A
    # rigorous SBC with many simulations is an offline job.
    assert 0.3 <= result.coverage_68 <= 1.0
    assert 0.6 <= result.coverage_95 <= 1.0


def test_run_emcee_sampler_rejects_bad_walker_count() -> None:
    prior = UniformPrior((ParameterSpec("spin", -0.9, 0.99),))
    forward = named_forward_model(
        _base_model, ["spin"], fixed={"eddington_ratio": 0.1, "f_col": 1.7}
    )
    data, variance = _noisy_data({"spin": 0.5}, 0.05, seed=3)
    problem = make_spectrum_posterior(prior, forward, data, variance)

    with pytest.raises(ValueError, match="walker"):
        run_emcee_sampler(
            problem,
            initial_positions=np.array([[0.5], [0.5], [0.5]]),
            draws=10,
        )


def test_make_spectrum_posterior_rejects_mismatched_shapes() -> None:
    prior = UniformPrior((ParameterSpec("spin", -0.9, 0.99),))
    forward = named_forward_model(_base_model, ["spin"])

    with pytest.raises(ValueError, match="same shape"):
        make_spectrum_posterior(
            prior,
            forward,
            np.ones(4, dtype=np.float64),
            np.ones(5, dtype=np.float64),
        )


def test_initial_walker_ball_stays_within_bounds() -> None:
    prior = UniformPrior(
        (
            ParameterSpec("spin", -0.9, 0.99),
            ParameterSpec("f_col", 1.4, 2.2),
        )
    )

    positions = initial_walker_ball(
        prior, np.array([0.98, 2.19]), walker_count=64, scale_fraction=0.2, seed=1
    )

    assert np.all(positions >= prior.lower_bounds)
    assert np.all(positions <= prior.upper_bounds)


def test_named_forward_model_requires_matching_dimension() -> None:
    forward = named_forward_model(_base_model, ["spin", "f_col"])

    with pytest.raises(ValueError, match="shape"):
        forward(np.array([0.5]))
