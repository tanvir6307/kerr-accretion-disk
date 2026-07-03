"""Multi-parameter joint spectral inference with an ensemble sampler.

This layer fits several disk parameters jointly (for example spin, Eddington
ratio, color correction, and inclination) with an affine-invariant ensemble
sampler, marginalizing nuisance parameters to obtain the spin posterior. It also
provides simulation-based calibration for the actual spectral forward model
rather than a one-dimensional analytic surrogate.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrdisk.inference import (
    ChainResult,
    PosteriorProblem,
    SimulationCalibrationResult,
    UniformPrior,
    credible_interval,
    simulation_based_calibration,
)

type FloatArray = NDArray[np.float64]
type ForwardModel = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class JointParameterSummary:
    """Marginal summary for one fitted parameter."""

    name: str
    mean: float
    median: float
    ci68_lower: float
    ci68_upper: float
    ci95_lower: float
    ci95_upper: float


@dataclass(frozen=True)
class JointPosteriorSummary:
    """Marginal summaries for all fitted parameters of a joint fit."""

    parameters: tuple[JointParameterSummary, ...]

    def by_name(self, name: str) -> JointParameterSummary:
        """Return the marginal summary for one parameter name."""

        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        msg = f"no fitted parameter named {name!r}"
        raise KeyError(msg)


def make_spectrum_posterior(
    prior: UniformPrior,
    forward_model: ForwardModel,
    observed: ArrayLike,
    variance: ArrayLike,
) -> PosteriorProblem:
    """Build a Gaussian-spectrum posterior over the fitted parameters.

    The variance is fixed by the data noise model, so nonpositive model bins are
    not rejected; only nonfinite model output or a forward-model failure yields a
    ``-inf`` log-likelihood.
    """

    observed_array = _as_1d("observed", observed)
    variance_array = _as_1d("variance", variance)
    if observed_array.shape != variance_array.shape:
        msg = "observed and variance must have the same shape"
        raise ValueError(msg)
    if np.any(variance_array <= 0.0):
        msg = "variance must be positive"
        raise ValueError(msg)
    log_norm = float(np.sum(np.log(2.0 * pi * variance_array)))

    def log_likelihood(parameters: FloatArray) -> float:
        try:
            model = np.asarray(forward_model(parameters), dtype=np.float64)
        except (FloatingPointError, ValueError):
            return float("-inf")
        if model.shape != observed_array.shape or not np.all(np.isfinite(model)):
            return float("-inf")
        residual = observed_array - model
        return -0.5 * (float(np.sum(residual * residual / variance_array)) + log_norm)

    return PosteriorProblem(prior=prior, log_likelihood=log_likelihood)


def initial_walker_ball(
    prior: UniformPrior,
    center: ArrayLike,
    *,
    walker_count: int,
    scale_fraction: float = 0.02,
    seed: int = 0,
) -> FloatArray:
    """Return ensemble walker start positions clustered near ``center``."""

    _validate_walker_count(walker_count, prior.dimension)
    if not isfinite(scale_fraction) or scale_fraction <= 0.0:
        msg = "scale_fraction must be finite and positive"
        raise ValueError(msg)
    if seed < 0:
        msg = "seed must be nonnegative"
        raise ValueError(msg)
    center_vector = _as_vector("center", center, prior.dimension)
    rng = np.random.default_rng(seed)
    scale = prior.widths * scale_fraction
    positions = center_vector + rng.normal(size=(walker_count, prior.dimension)) * scale
    margin = 1.0e-9 * prior.widths
    return np.clip(
        positions,
        prior.lower_bounds + margin,
        prior.upper_bounds - margin,
    )


def run_emcee_sampler(
    problem: PosteriorProblem,
    *,
    initial_positions: ArrayLike,
    draws: int,
    burn_in: int = 0,
    seed: int = 0,
) -> ChainResult:
    """Run an affine-invariant ensemble sampler and return chain samples."""

    import emcee

    positions = np.asarray(initial_positions, dtype=np.float64)
    dimension = problem.prior.dimension
    if positions.ndim != 2 or positions.shape[1] != dimension:
        msg = f"initial_positions must have shape (walkers, {dimension})"
        raise ValueError(msg)
    walker_count = int(positions.shape[0])
    _validate_walker_count(walker_count, dimension)
    if draws < 1:
        msg = "draws must be positive"
        raise ValueError(msg)
    if burn_in < 0:
        msg = "burn_in must be nonnegative"
        raise ValueError(msg)
    if seed < 0:
        msg = "seed must be nonnegative"
        raise ValueError(msg)
    initial_log_prob = np.array(
        [problem.log_posterior(position) for position in positions],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(initial_log_prob)):
        msg = "all initial_positions must have finite posterior probability"
        raise ValueError(msg)

    sampler = emcee.EnsembleSampler(
        walker_count,
        dimension,
        problem.log_posterior,
    )
    sampler._random = np.random.RandomState(seed)  # noqa: SLF001
    sampler.run_mcmc(positions, draws + burn_in, progress=False)

    chain = np.asarray(sampler.get_chain(), dtype=np.float64)
    log_prob = np.asarray(sampler.get_log_prob(), dtype=np.float64)
    samples = np.ascontiguousarray(
        np.transpose(chain[burn_in:], (1, 0, 2)),
        dtype=np.float64,
    )
    log_probability = np.ascontiguousarray(
        log_prob[burn_in:].T,
        dtype=np.float64,
    )
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=np.float64)
    return ChainResult(
        samples=samples,
        log_probability=log_probability,
        acceptance_fraction=acceptance,
    )


def flatten_chain(chain_result: ChainResult) -> FloatArray:
    """Return chain samples flattened to shape (draws * walkers, parameters)."""

    samples = chain_result.samples
    return samples.reshape(-1, samples.shape[2])


def summarize_joint_posterior(
    prior: UniformPrior,
    chain_result: ChainResult,
) -> JointPosteriorSummary:
    """Return marginal summaries for every fitted parameter."""

    flat = flatten_chain(chain_result)
    summaries: list[JointParameterSummary] = []
    for index, spec in enumerate(prior.parameters):
        column = flat[:, index]
        ci68 = credible_interval(column, 0.68)
        ci95 = credible_interval(column, 0.95)
        summaries.append(
            JointParameterSummary(
                name=spec.name,
                mean=float(np.mean(column)),
                median=float(np.median(column)),
                ci68_lower=ci68[0],
                ci68_upper=ci68[1],
                ci95_lower=ci95[0],
                ci95_upper=ci95[1],
            )
        )
    return JointPosteriorSummary(parameters=tuple(summaries))


def run_spectrum_sbc(
    prior: UniformPrior,
    forward_model: ForwardModel,
    *,
    rank_parameter: str,
    relative_error: float,
    simulations: int,
    walker_count: int,
    draws: int,
    burn_in: int,
    seed: int,
    rank_bins: int = 10,
) -> SimulationCalibrationResult:
    """Run simulation-based calibration on the spectral forward model.

    For each simulation a truth is drawn from the prior, a noisy spectrum is
    generated, and the joint posterior is sampled. The rank of the true value of
    ``rank_parameter`` within its marginal posterior is accumulated across
    simulations. Ensemble walkers are dispersed across the prior rather than
    started at the truth, so the ranks are an honest calibration check.
    """

    rank_index = _parameter_index(prior, rank_parameter)
    if not isfinite(relative_error) or relative_error <= 0.0:
        msg = "relative_error must be finite and positive"
        raise ValueError(msg)
    if simulations < 1:
        msg = "simulations must be positive"
        raise ValueError(msg)
    _validate_walker_count(walker_count, prior.dimension)
    if draws < 1 or burn_in < 0:
        msg = "draws must be positive and burn_in nonnegative"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    truths: list[float] = []
    posterior_rows: list[FloatArray] = []
    for _ in range(simulations):
        truth, expectation = _draw_valid_truth(prior, forward_model, rng)
        sigma = relative_error * expectation
        variance = sigma * sigma
        data = rng.normal(loc=expectation, scale=sigma)
        problem = make_spectrum_posterior(prior, forward_model, data, variance)
        positions = prior.sample(rng, walker_count)
        chain = run_emcee_sampler(
            problem,
            initial_positions=positions,
            draws=draws,
            burn_in=burn_in,
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        posterior_rows.append(flatten_chain(chain)[:, rank_index])
        truths.append(float(truth[rank_index]))

    length = min(row.size for row in posterior_rows)
    posterior_samples = np.array([row[:length] for row in posterior_rows])
    return simulation_based_calibration(
        np.array(truths, dtype=np.float64),
        posterior_samples,
        rank_bins=rank_bins,
    )


def named_forward_model(
    base_model: Callable[[Mapping[str, float]], ArrayLike],
    parameter_names: Sequence[str],
    *,
    fixed: Mapping[str, float] | None = None,
) -> ForwardModel:
    """Adapt a keyword spectral model to a positional parameter vector.

    ``parameter_names`` maps the sampled parameter vector to named arguments;
    ``fixed`` supplies additional named arguments held constant during the fit.
    """

    names = tuple(parameter_names)
    if not names:
        msg = "parameter_names must be nonempty"
        raise ValueError(msg)
    constant = dict(fixed or {})

    def model(parameters: FloatArray) -> ArrayLike:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (len(names),):
            msg = f"parameters must have shape ({len(names)},)"
            raise ValueError(msg)
        arguments = dict(constant)
        arguments.update(dict(zip(names, (float(v) for v in values), strict=True)))
        return base_model(arguments)

    return model


def _draw_valid_truth(
    prior: UniformPrior,
    forward_model: ForwardModel,
    rng: np.random.Generator,
    *,
    max_attempts: int = 64,
) -> tuple[FloatArray, FloatArray]:
    """Draw a prior truth whose forward model evaluates to a finite spectrum.

    Rare prior draws can land on a parameter combination the forward model
    cannot evaluate (for example a numerically degenerate flux). Such draws are
    resampled so a single pathological point does not abort the whole
    calibration run.
    """

    for _ in range(max_attempts):
        truth = prior.sample(rng, 1)[0]
        try:
            expectation = np.asarray(forward_model(truth), dtype=np.float64)
        except (FloatingPointError, ValueError):
            continue
        if np.all(np.isfinite(expectation)) and np.any(expectation > 0.0):
            return truth, expectation
    msg = "could not draw a valid SBC truth from the prior forward model"
    raise RuntimeError(msg)


def _parameter_index(prior: UniformPrior, name: str) -> int:
    for index, spec in enumerate(prior.parameters):
        if spec.name == name:
            return index
    msg = f"prior has no parameter named {name!r}"
    raise KeyError(msg)


def _validate_walker_count(walker_count: int, dimension: int) -> None:
    if walker_count < 2 * dimension:
        msg = "walker_count must be at least twice the parameter dimension"
        raise ValueError(msg)
    if walker_count % 2 != 0:
        msg = "walker_count must be even for the ensemble sampler"
        raise ValueError(msg)


def _as_1d(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _as_vector(name: str, values: ArrayLike, dimension: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension,):
        msg = f"{name} must have shape ({dimension},)"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array
