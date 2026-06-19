"""Tests for inference adapters and calibration utilities."""

import csv
from math import log, sqrt
from pathlib import Path

import numpy as np
import pytest

from kerrdisk.inference import (
    ParameterSpec,
    PosteriorProblem,
    UniformPrior,
    credible_interval,
    effective_sample_size,
    optimize_posterior,
    posterior_rank,
    prior_predictive_check,
    run_random_walk_metropolis,
    simulation_based_calibration,
    split_rhat,
    summarize_chains,
)


def _one_dimensional_prior() -> UniformPrior:
    return UniformPrior((ParameterSpec("x", -5.0, 5.0),))


def _standard_normal_problem() -> PosteriorProblem:
    prior = _one_dimensional_prior()

    def log_likelihood(parameters: np.ndarray) -> float:
        return -0.5 * float(parameters[0] * parameters[0])

    return PosteriorProblem(prior=prior, log_likelihood=log_likelihood)


def test_uniform_prior_transform_and_log_density() -> None:
    prior = UniformPrior(
        (
            ParameterSpec("spin", -0.998, 0.998),
            ParameterSpec("f_col", 1.4, 2.2),
        )
    )

    parameters = prior.transform_unit_cube(np.array([0.5, 0.25]))
    unit = prior.inverse_transform(parameters)

    assert parameters == pytest.approx(np.array([0.0, 1.6]))
    assert unit == pytest.approx(np.array([0.5, 0.25]))
    assert prior.log_prior(parameters) == pytest.approx(
        -log((2.0 * 0.998) * (2.2 - 1.4))
    )
    assert prior.log_prior(np.array([2.0, 1.6])) == float("-inf")


def test_prior_rejects_invalid_configuration_and_values() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        ParameterSpec("", 0.0, 1.0)
    with pytest.raises(ValueError, match="less than"):
        ParameterSpec("x", 1.0, 1.0)
    with pytest.raises(ValueError, match="at least one"):
        UniformPrior(())

    prior = _one_dimensional_prior()
    with pytest.raises(ValueError, match="shape"):
        prior.transform_unit_cube(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prior.transform_unit_cube(np.array([1.5]))
    with pytest.raises(ValueError, match="count"):
        prior.sample(np.random.default_rng(1), 0)


def test_posterior_problem_returns_negative_infinity_for_invalid_regions() -> None:
    problem = _standard_normal_problem()

    assert problem.log_posterior(np.array([10.0])) == float("-inf")
    assert problem.log_posterior(np.array([0.0])) > float("-inf")


def test_optimizer_recovers_gaussian_mode() -> None:
    problem = _standard_normal_problem()

    result = optimize_posterior(
        problem,
        starts=np.array([[-4.0], [3.0]]),
        tolerance=1.0e-8,
    )

    assert result.success
    assert result.parameters[0] == pytest.approx(0.0, abs=1.0e-5)


def test_optimizer_rejects_invalid_controls() -> None:
    problem = _standard_normal_problem()

    with pytest.raises(ValueError, match="initial_step_fraction"):
        optimize_posterior(problem, initial_step_fraction=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        optimize_posterior(problem, tolerance=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        optimize_posterior(problem, max_iterations=0)
    with pytest.raises(ValueError, match="starts"):
        optimize_posterior(problem, starts=np.ones((2, 2)))


def test_random_walk_metropolis_samples_standard_normal() -> None:
    problem = _standard_normal_problem()
    initial_positions = np.array([[-0.2], [0.1], [0.3], [-0.1]])

    chains = run_random_walk_metropolis(
        problem,
        initial_positions=initial_positions,
        step_scale=np.array([0.8]),
        draws=2_000,
        burn_in=500,
        seed=123,
    )
    diagnostics = summarize_chains(chains, rhat_max=1.08, ess_min=100.0)
    flattened = chains.samples[:, :, 0].reshape(-1)

    assert diagnostics.converged
    assert split_rhat(chains.samples)[0] < 1.08
    assert effective_sample_size(chains.samples)[0] > 100.0
    assert np.mean(chains.acceptance_fraction) == pytest.approx(0.75, abs=0.15)
    assert flattened.mean() == pytest.approx(0.0, abs=0.08)
    assert flattened.std(ddof=1) == pytest.approx(1.0, abs=0.1)


def test_random_walk_metropolis_rejects_invalid_inputs() -> None:
    problem = _standard_normal_problem()
    initial_positions = np.array([[0.0]])

    with pytest.raises(ValueError, match="draws"):
        run_random_walk_metropolis(
            problem,
            initial_positions=initial_positions,
            step_scale=0.1,
            draws=0,
        )
    with pytest.raises(ValueError, match="burn_in"):
        run_random_walk_metropolis(
            problem,
            initial_positions=initial_positions,
            step_scale=0.1,
            draws=10,
            burn_in=-1,
        )
    with pytest.raises(ValueError, match="step_scale"):
        run_random_walk_metropolis(
            problem,
            initial_positions=initial_positions,
            step_scale=0.0,
            draws=10,
        )
    with pytest.raises(ValueError, match="finite posterior"):
        run_random_walk_metropolis(
            problem,
            initial_positions=np.array([[10.0]]),
            step_scale=0.1,
            draws=10,
        )


def test_credible_interval_and_rank() -> None:
    samples = np.arange(100.0)

    lower, upper = credible_interval(samples, 0.68)

    assert lower == pytest.approx(15.84)
    assert upper == pytest.approx(83.16)
    assert posterior_rank(samples, 50.0) == 50


def test_diagnostics_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        split_rhat(np.ones((10, 1)))
    with pytest.raises(ValueError, match="finite"):
        credible_interval([1.0, float("nan")], 0.68)
    with pytest.raises(ValueError, match="between"):
        credible_interval([1.0], 1.0)
    with pytest.raises(ValueError, match="true_value"):
        posterior_rank([1.0], float("nan"))


def test_prior_predictive_check_records_finite_output_summary() -> None:
    prior = _one_dimensional_prior()

    result = prior_predictive_check(
        prior,
        lambda parameter: np.array([parameter[0], parameter[0] ** 2]),
        sample_count=20,
        seed=12,
    )

    assert result.parameters.shape == (20, 1)
    assert result.output_mean.shape == (2,)
    assert np.all(result.output_min <= result.output_max)


def test_prior_predictive_check_rejects_bad_simulator_output() -> None:
    prior = _one_dimensional_prior()

    with pytest.raises(ValueError, match="nonempty"):

        def empty_output(parameter: np.ndarray) -> np.ndarray:
            _ = parameter
            return np.array([])

        prior_predictive_check(
            prior,
            empty_output,
            sample_count=2,
            seed=1,
        )
    with pytest.raises(ValueError, match="consistent"):
        prior_predictive_check(
            prior,
            lambda parameter: np.ones(1 if parameter[0] < 0.0 else 2),
            sample_count=10,
            seed=1,
        )


def test_simulation_based_calibration_for_calibrated_gaussian_posterior() -> None:
    rng = np.random.default_rng(123)
    simulation_count = 300
    draw_count = 400
    true_values = rng.normal(size=simulation_count)
    posterior_samples = rng.normal(size=(simulation_count, draw_count))

    result = simulation_based_calibration(
        true_values,
        posterior_samples,
        rank_bins=10,
    )

    assert result.ranks.shape == (simulation_count,)
    assert result.rank_histogram.sum() == simulation_count
    assert result.coverage_68 == pytest.approx(0.68, abs=0.06)
    assert result.coverage_95 == pytest.approx(0.95, abs=0.04)
    assert result.max_rank_histogram_z < 3.0


def test_simulation_based_calibration_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        simulation_based_calibration([0.0], np.ones(2))
    with pytest.raises(ValueError, match="one value"):
        simulation_based_calibration([0.0, 1.0], np.ones((1, 3)))
    with pytest.raises(ValueError, match="at least two"):
        simulation_based_calibration([0.0], np.ones((1, 1)))
    with pytest.raises(ValueError, match="rank_bins"):
        simulation_based_calibration([0.0], np.ones((1, 3)), rank_bins=0)


def test_summarize_chains_rejects_bad_thresholds() -> None:
    problem = _standard_normal_problem()
    chains = run_random_walk_metropolis(
        problem,
        initial_positions=np.array([[-0.1], [0.1]]),
        step_scale=0.5,
        draws=10,
        seed=1,
    )

    with pytest.raises(ValueError, match="rhat_max"):
        summarize_chains(chains, rhat_max=1.0)
    with pytest.raises(ValueError, match="ess_min"):
        summarize_chains(chains, ess_min=0.0)


def test_effective_sample_size_for_independent_draws_is_large() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(4, 500, 1))

    ess = effective_sample_size(samples)

    assert ess[0] > 1_500.0
    assert ess[0] <= 2_000.0


def test_split_rhat_detects_disagreeing_chains() -> None:
    rng = np.random.default_rng(4)
    first = rng.normal(loc=-2.0, scale=0.1, size=(2, 200, 1))
    second = rng.normal(loc=2.0, scale=0.1, size=(2, 200, 1))
    samples = np.concatenate([first, second], axis=0)

    assert split_rhat(samples)[0] > sqrt(2.0)


def test_frozen_phase10_prior_table_has_valid_bounds() -> None:
    path = Path("paper/tables/priors.csv")

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert {row["status"] for row in rows} == {"frozen_phase10"}
    for row in rows:
        assert float(row["lower"]) < float(row["upper"])
        assert row["prior"] == "uniform"
