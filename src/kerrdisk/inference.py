"""Inference adapters, diagnostics, and calibration utilities."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type LogLikelihood = Callable[[FloatArray], float]
type Simulator = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded scalar parameter."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name:
            msg = "parameter name must be nonempty"
            raise ValueError(msg)
        if not isfinite(self.lower) or not isfinite(self.upper):
            msg = "parameter bounds must be finite"
            raise ValueError(msg)
        if self.lower >= self.upper:
            msg = "parameter lower bound must be less than upper bound"
            raise ValueError(msg)


@dataclass(frozen=True)
class UniformPrior:
    """Independent bounded uniform prior."""

    parameters: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            msg = "prior must contain at least one parameter"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        """Return the number of parameters."""

        return len(self.parameters)

    @property
    def lower_bounds(self) -> FloatArray:
        """Return lower bounds as an array."""

        return np.array([parameter.lower for parameter in self.parameters])

    @property
    def upper_bounds(self) -> FloatArray:
        """Return upper bounds as an array."""

        return np.array([parameter.upper for parameter in self.parameters])

    @property
    def widths(self) -> FloatArray:
        """Return prior widths."""

        return self.upper_bounds - self.lower_bounds

    def transform_unit_cube(self, unit_values: ArrayLike) -> FloatArray:
        """Map unit-cube coordinates to physical parameters."""

        values = _as_vector("unit_values", unit_values, self.dimension)
        if np.any((values < 0.0) | (values > 1.0)):
            msg = "unit_values must lie inside [0, 1]"
            raise ValueError(msg)
        return self.lower_bounds + (values * self.widths)

    def inverse_transform(self, parameters: ArrayLike) -> FloatArray:
        """Map physical parameters to unit-cube coordinates."""

        values = _as_vector("parameters", parameters, self.dimension)
        return (values - self.lower_bounds) / self.widths

    def contains(self, parameters: ArrayLike) -> bool:
        """Return whether all parameters lie inside the prior bounds."""

        try:
            values = _as_vector("parameters", parameters, self.dimension)
        except ValueError:
            return False
        return bool(
            np.all(values >= self.lower_bounds) and np.all(values <= self.upper_bounds)
        )

    def log_prior(self, parameters: ArrayLike) -> float:
        """Return independent uniform log-prior density."""

        if not self.contains(parameters):
            return float("-inf")
        return -float(np.sum(np.log(self.widths)))

    def sample(self, rng: np.random.Generator, count: int) -> FloatArray:
        """Draw prior samples."""

        if count < 1:
            msg = "count must be positive"
            raise ValueError(msg)
        unit = rng.uniform(size=(count, self.dimension))
        return self.lower_bounds + (unit * self.widths)


@dataclass(frozen=True)
class PosteriorProblem:
    """Prior and likelihood defining an unnormalized posterior."""

    prior: UniformPrior
    log_likelihood: LogLikelihood

    def log_posterior(self, parameters: ArrayLike) -> float:
        """Return log likelihood plus log prior."""

        values = _as_vector("parameters", parameters, self.prior.dimension)
        prior_value = self.prior.log_prior(values)
        if not isfinite(prior_value):
            return float("-inf")
        likelihood_value = self.log_likelihood(values)
        if not isfinite(likelihood_value):
            return float("-inf")
        return prior_value + float(likelihood_value)


@dataclass(frozen=True)
class OptimizationResult:
    """Deterministic posterior optimization result."""

    parameters: FloatArray
    log_posterior: float
    success: bool
    iterations: int
    message: str


@dataclass(frozen=True)
class ChainResult:
    """MCMC chain samples and diagnostics."""

    samples: FloatArray
    log_probability: FloatArray
    acceptance_fraction: FloatArray


@dataclass(frozen=True)
class ChainDiagnostics:
    """Summary convergence diagnostics for a chain result."""

    rhat: FloatArray
    ess: FloatArray
    acceptance_fraction: FloatArray
    converged: bool


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Finite-output prior predictive check result."""

    parameters: FloatArray
    output_mean: FloatArray
    output_min: FloatArray
    output_max: FloatArray


@dataclass(frozen=True)
class SimulationCalibrationResult:
    """One-dimensional SBC rank and coverage summary."""

    ranks: NDArray[np.int64]
    rank_histogram: NDArray[np.int64]
    expected_per_bin: float
    coverage_68: float
    coverage_95: float
    max_rank_histogram_z: float


def _as_vector(name: str, values: ArrayLike, dimension: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension,):
        msg = f"{name} must have shape ({dimension},)"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _as_chain_array(samples: ArrayLike) -> FloatArray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 3:
        msg = "samples must have shape (chains, draws, parameters)"
        raise ValueError(msg)
    if array.shape[0] < 2 or array.shape[1] < 4 or array.shape[2] < 1:
        msg = "samples must contain at least 2 chains, 4 draws, and 1 parameter"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "samples must contain only finite values"
        raise ValueError(msg)
    return array


def optimize_posterior(
    problem: PosteriorProblem,
    *,
    starts: ArrayLike | None = None,
    initial_step_fraction: float = 0.25,
    tolerance: float = 1.0e-6,
    max_iterations: int = 512,
) -> OptimizationResult:
    """Maximize a posterior with deterministic coordinate-pattern search."""

    if not isfinite(initial_step_fraction) or initial_step_fraction <= 0.0:
        msg = "initial_step_fraction must be finite and positive"
        raise ValueError(msg)
    if not isfinite(tolerance) or tolerance <= 0.0:
        msg = "tolerance must be finite and positive"
        raise ValueError(msg)
    if max_iterations < 1:
        msg = "max_iterations must be positive"
        raise ValueError(msg)

    start_array = _optimization_starts(problem.prior, starts)
    best_parameters = start_array[0].copy()
    best_log_post = float("-inf")
    total_iterations = 0
    for start in start_array:
        result = _optimize_single_start(
            problem,
            start,
            initial_step_fraction=initial_step_fraction,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        total_iterations += result.iterations
        if result.log_posterior > best_log_post:
            best_parameters = result.parameters
            best_log_post = result.log_posterior

    success = isfinite(best_log_post)
    return OptimizationResult(
        parameters=best_parameters,
        log_posterior=best_log_post,
        success=success,
        iterations=total_iterations,
        message="coordinate-pattern search" if success else "no finite posterior",
    )


def _optimization_starts(prior: UniformPrior, starts: ArrayLike | None) -> FloatArray:
    if starts is None:
        return ((prior.lower_bounds + prior.upper_bounds) / 2.0).reshape(
            1,
            prior.dimension,
        )
    array = np.asarray(starts, dtype=np.float64)
    if array.shape == (prior.dimension,):
        array = array.reshape(1, prior.dimension)
    if array.ndim != 2 or array.shape[1] != prior.dimension:
        msg = f"starts must have shape (n, {prior.dimension})"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "starts must contain only finite values"
        raise ValueError(msg)
    return array


def _optimize_single_start(
    problem: PosteriorProblem,
    start: FloatArray,
    *,
    initial_step_fraction: float,
    tolerance: float,
    max_iterations: int,
) -> OptimizationResult:
    current = np.minimum(
        np.maximum(start.copy(), problem.prior.lower_bounds),
        problem.prior.upper_bounds,
    )
    current_log_post = problem.log_posterior(current)
    step = problem.prior.widths * initial_step_fraction
    iterations = 0
    while float(np.max(step)) > tolerance and iterations < max_iterations:
        improved = False
        for parameter_index in range(problem.prior.dimension):
            for sign in (1.0, -1.0):
                proposal = current.copy()
                proposal[parameter_index] += sign * step[parameter_index]
                proposal_log_post = problem.log_posterior(proposal)
                if proposal_log_post > current_log_post:
                    current = proposal
                    current_log_post = proposal_log_post
                    improved = True
        if not improved:
            step *= 0.5
        iterations += 1
    return OptimizationResult(
        parameters=current,
        log_posterior=current_log_post,
        success=isfinite(current_log_post),
        iterations=iterations,
        message="single-start coordinate-pattern search",
    )


def run_random_walk_metropolis(
    problem: PosteriorProblem,
    *,
    initial_positions: ArrayLike,
    step_scale: ArrayLike,
    draws: int,
    burn_in: int = 0,
    seed: int = 0,
) -> ChainResult:
    """Run independent random-walk Metropolis chains."""

    if draws < 1:
        msg = "draws must be positive"
        raise ValueError(msg)
    if burn_in < 0:
        msg = "burn_in must be nonnegative"
        raise ValueError(msg)
    if seed < 0:
        msg = "seed must be nonnegative"
        raise ValueError(msg)
    positions = np.asarray(initial_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != problem.prior.dimension:
        msg = f"initial_positions must have shape (chains, {problem.prior.dimension})"
        raise ValueError(msg)
    if positions.shape[0] < 1:
        msg = "initial_positions must contain at least one chain"
        raise ValueError(msg)
    if not np.all(np.isfinite(positions)):
        msg = "initial_positions must contain only finite values"
        raise ValueError(msg)
    steps = _broadcast_step_scale(step_scale, problem.prior.dimension)
    rng = np.random.default_rng(seed)

    chain_count = positions.shape[0]
    current = positions.copy()
    current_log_prob = np.array(
        [problem.log_posterior(position) for position in current],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(current_log_prob)):
        msg = "all initial_positions must have finite posterior probability"
        raise ValueError(msg)

    samples = np.empty((chain_count, draws, problem.prior.dimension), dtype=np.float64)
    log_probability = np.empty((chain_count, draws), dtype=np.float64)
    accepted = np.zeros(chain_count, dtype=np.float64)
    total_steps = draws + burn_in
    for step_index in range(total_steps):
        for chain_index in range(chain_count):
            proposal = current[chain_index] + rng.normal(scale=steps)
            proposal_log_prob = problem.log_posterior(proposal)
            log_acceptance = proposal_log_prob - current_log_prob[chain_index]
            if isfinite(log_acceptance) and log(rng.uniform()) < min(
                0.0,
                log_acceptance,
            ):
                current[chain_index] = proposal
                current_log_prob[chain_index] = proposal_log_prob
                if step_index >= burn_in:
                    accepted[chain_index] += 1.0
        if step_index >= burn_in:
            draw_index = step_index - burn_in
            samples[:, draw_index, :] = current
            log_probability[:, draw_index] = current_log_prob

    return ChainResult(
        samples=samples,
        log_probability=log_probability,
        acceptance_fraction=accepted / draws,
    )


def _broadcast_step_scale(step_scale: ArrayLike, dimension: int) -> FloatArray:
    array = np.asarray(step_scale, dtype=np.float64)
    if array.shape == ():
        array = np.full(dimension, float(array), dtype=np.float64)
    if array.shape != (dimension,):
        msg = f"step_scale must be scalar or have shape ({dimension},)"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        msg = "step_scale must be finite and positive"
        raise ValueError(msg)
    return array


def split_rhat(samples: ArrayLike) -> FloatArray:
    """Return basic split-chain R-hat for each parameter."""

    chains = _split_chains(_as_chain_array(samples))
    chain_count, draw_count, parameter_count = chains.shape
    values = np.empty(parameter_count, dtype=np.float64)
    for parameter_index in range(parameter_count):
        parameter_samples = chains[:, :, parameter_index]
        chain_means = np.mean(parameter_samples, axis=1)
        chain_variances = np.var(parameter_samples, axis=1, ddof=1)
        within = float(np.mean(chain_variances))
        between = draw_count * float(np.var(chain_means, ddof=1))
        if within == 0.0:
            values[parameter_index] = float("inf")
            continue
        variance_hat = (((draw_count - 1.0) / draw_count) * within) + (
            between / draw_count
        )
        values[parameter_index] = sqrt(variance_hat / within)
    _ = chain_count
    return values


def _split_chains(samples: FloatArray) -> FloatArray:
    half_draws = samples.shape[1] // 2
    if half_draws < 2:
        msg = "each split chain must contain at least two draws"
        raise ValueError(msg)
    trimmed = samples[:, : 2 * half_draws, :]
    first_half = trimmed[:, :half_draws, :]
    second_half = trimmed[:, half_draws:, :]
    return np.concatenate([first_half, second_half], axis=0)


def effective_sample_size(samples: ArrayLike) -> FloatArray:
    """Estimate ESS with an initial-positive autocorrelation sequence."""

    chains = _as_chain_array(samples)
    chain_count, draw_count, parameter_count = chains.shape
    ess = np.empty(parameter_count, dtype=np.float64)
    total_draws = chain_count * draw_count
    for parameter_index in range(parameter_count):
        rho_sum = 0.0
        for lag in range(1, draw_count):
            rho = _mean_chain_autocorrelation(chains[:, :, parameter_index], lag)
            if rho <= 0.0:
                break
            rho_sum += rho
        ess[parameter_index] = min(total_draws / (1.0 + (2.0 * rho_sum)), total_draws)
    return ess


def _mean_chain_autocorrelation(parameter_samples: FloatArray, lag: int) -> float:
    correlations: list[float] = []
    for chain in parameter_samples:
        centered = chain - np.mean(chain)
        denominator = float(centered @ centered)
        if denominator == 0.0:
            correlations.append(0.0)
            continue
        numerator = float(centered[:-lag] @ centered[lag:])
        correlations.append(numerator / denominator)
    return float(np.mean(correlations))


def summarize_chains(
    chain_result: ChainResult,
    *,
    rhat_max: float = 1.05,
    ess_min: float = 100.0,
) -> ChainDiagnostics:
    """Return convergence diagnostics for a chain result."""

    if not isfinite(rhat_max) or rhat_max <= 1.0:
        msg = "rhat_max must be finite and greater than 1"
        raise ValueError(msg)
    if not isfinite(ess_min) or ess_min <= 0.0:
        msg = "ess_min must be finite and positive"
        raise ValueError(msg)
    rhat = split_rhat(chain_result.samples)
    ess = effective_sample_size(chain_result.samples)
    converged = bool(np.all(rhat <= rhat_max) and np.all(ess >= ess_min))
    return ChainDiagnostics(
        rhat=rhat,
        ess=ess,
        acceptance_fraction=chain_result.acceptance_fraction.copy(),
        converged=converged,
    )


def credible_interval(samples: ArrayLike, level: float) -> tuple[float, float]:
    """Return a scalar equal-tailed credible interval."""

    if not isfinite(level) or not 0.0 < level < 1.0:
        msg = "level must be between 0 and 1"
        raise ValueError(msg)
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)):
        msg = "samples must contain finite values"
        raise ValueError(msg)
    alpha = (1.0 - level) / 2.0
    interval = np.quantile(values, [alpha, 1.0 - alpha])
    return float(interval[0]), float(interval[1])


def posterior_rank(samples: ArrayLike, true_value: float) -> int:
    """Return count of posterior draws below the true scalar value."""

    if not isfinite(true_value):
        msg = "true_value must be finite"
        raise ValueError(msg)
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)):
        msg = "samples must contain finite values"
        raise ValueError(msg)
    return int(np.sum(values < true_value))


def prior_predictive_check(
    prior: UniformPrior,
    simulator: Simulator,
    *,
    sample_count: int,
    seed: int,
) -> PriorPredictiveResult:
    """Draw prior samples and require finite simulator outputs."""

    if seed < 0:
        msg = "seed must be nonnegative"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    parameters = prior.sample(rng, sample_count)
    outputs = []
    for parameter in parameters:
        output = np.asarray(simulator(parameter), dtype=np.float64)
        if output.size < 1 or not np.all(np.isfinite(output)):
            msg = "simulator must return finite nonempty outputs"
            raise ValueError(msg)
        outputs.append(output.reshape(-1))
    output_array = _stack_equal_length_outputs(outputs)
    return PriorPredictiveResult(
        parameters=parameters,
        output_mean=np.mean(output_array, axis=0),
        output_min=np.min(output_array, axis=0),
        output_max=np.max(output_array, axis=0),
    )


def _stack_equal_length_outputs(outputs: Sequence[FloatArray]) -> FloatArray:
    length = outputs[0].size
    if any(output.size != length for output in outputs):
        msg = "simulator outputs must have consistent flattened length"
        raise ValueError(msg)
    return np.vstack(outputs)


def simulation_based_calibration(
    true_values: ArrayLike,
    posterior_samples: ArrayLike,
    *,
    rank_bins: int = 10,
) -> SimulationCalibrationResult:
    """Summarize one-dimensional SBC ranks and 68/95 percent coverage."""

    truth = np.asarray(true_values, dtype=np.float64).reshape(-1)
    samples = np.asarray(posterior_samples, dtype=np.float64)
    if samples.ndim != 2:
        msg = "posterior_samples must have shape (simulations, draws)"
        raise ValueError(msg)
    if truth.shape != (samples.shape[0],):
        msg = "true_values must contain one value per simulation"
        raise ValueError(msg)
    if samples.shape[1] < 2:
        msg = "posterior_samples must contain at least two draws"
        raise ValueError(msg)
    if rank_bins < 1:
        msg = "rank_bins must be positive"
        raise ValueError(msg)
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(samples)):
        msg = "true_values and posterior_samples must be finite"
        raise ValueError(msg)

    ranks = np.array(
        [
            posterior_rank(draws, truth_value)
            for truth_value, draws in zip(truth, samples, strict=True)
        ],
        dtype=np.int64,
    )
    histogram = np.histogram(ranks, bins=rank_bins, range=(0, samples.shape[1]))[0]
    expected = float(ranks.size / rank_bins)
    max_rank_z = (
        0.0
        if expected == 0.0
        else float(np.max(np.abs(histogram - expected)) / sqrt(expected))
    )
    coverage_68 = _coverage_fraction(truth, samples, 0.68)
    coverage_95 = _coverage_fraction(truth, samples, 0.95)
    return SimulationCalibrationResult(
        ranks=ranks,
        rank_histogram=histogram.astype(np.int64),
        expected_per_bin=expected,
        coverage_68=coverage_68,
        coverage_95=coverage_95,
        max_rank_histogram_z=max_rank_z,
    )


def _coverage_fraction(
    true_values: FloatArray,
    posterior_samples: FloatArray,
    level: float,
) -> float:
    hits = []
    for truth, draws in zip(true_values, posterior_samples, strict=True):
        lower, upper = credible_interval(draws, level)
        hits.append(lower <= truth <= upper)
    return float(np.mean(hits))
