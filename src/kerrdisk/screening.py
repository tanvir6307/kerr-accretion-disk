"""Phase 11 screening-campaign utilities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from kerrdisk.likelihood import gaussian_log_likelihood
from kerrdisk.synthetic import (
    EnergyBins,
    GaussianRelativeNoise,
    SyntheticEpoch,
    derive_seed,
    generate_gaussian_epoch,
    make_log_energy_bins,
)

type FloatArray = NDArray[np.float64]
type ConditionStatus = Literal["COMPLETED", "FAILED"]


@dataclass(frozen=True)
class ScreeningThresholds:
    """Refinement thresholds for the coarse screening campaign."""

    high_bias_abs: float = 0.05
    low_identifiability_width_68: float = 0.25
    chi2_per_dof_max: float = 2.0
    failure_rate_max: float = 0.0


@dataclass(frozen=True)
class ScreeningConfig:
    """Versioned Phase 11 screening configuration."""

    config_version: str
    master_seed: int
    spins: tuple[float, ...]
    inclinations_deg: tuple[float, ...]
    eddington_ratios: tuple[float, ...]
    f_col_true_values: tuple[float, ...]
    f_col_fit: float
    inner_stress_delta_eta: tuple[float, ...]
    replicate_count: int
    energy_min_kev: float
    energy_max_kev: float
    energy_bin_count: int
    spin_grid_count: int
    gaussian_relative_error: float
    thresholds: ScreeningThresholds


@dataclass(frozen=True)
class ScreeningCondition:
    """One coarse-grid screening condition."""

    condition_id: str
    spin_true: float
    inclination_deg: float
    eddington_ratio: float
    f_col_true: float
    f_col_fit: float
    inner_stress_delta_eta: float
    replicate_count: int


@dataclass(frozen=True)
class ReplicateResult:
    """One screening replicate fit result."""

    condition_id: str
    replicate_index: int
    status: ConditionStatus
    failure_cause: str
    spin_true: float
    spin_map: float
    spin_mean: float
    bias: float
    ci68_lower: float
    ci68_upper: float
    ci95_lower: float
    ci95_upper: float
    chi2_per_dof: float
    runtime_s: float
    noise_seed: int


@dataclass(frozen=True)
class ConditionSummary:
    """Condition-level screening summary."""

    condition_id: str
    status: ConditionStatus
    failure_cause: str
    planned_replicates: int
    completed_replicates: int
    failed_replicates: int
    failure_rate: float
    spin_true: float
    inclination_deg: float
    eddington_ratio: float
    f_col_true: float
    f_col_fit: float
    inner_stress_delta_eta: float
    mean_bias: float
    median_bias: float
    rmse: float
    mean_width_68: float
    mean_width_95: float
    coverage_68: float
    coverage_95: float
    mean_chi2_per_dof: float
    mean_runtime_s: float
    high_bias_flag: bool
    low_identifiability_flag: bool
    poor_fit_flag: bool
    needs_refinement: bool


@dataclass(frozen=True)
class ScreeningCampaignResult:
    """Full Phase 11 screening-campaign result."""

    config: ScreeningConfig
    conditions: tuple[ScreeningCondition, ...]
    replicates: tuple[ReplicateResult, ...]
    summaries: tuple[ConditionSummary, ...]
    confirmatory_conditions: tuple[ConditionSummary, ...]


def default_screening_config() -> ScreeningConfig:
    """Return the default lightweight Phase 11 screening configuration."""

    return ScreeningConfig(
        config_version="phase11_screening_v1",
        master_seed=20260619,
        spins=(-0.5, 0.0, 0.5, 0.9),
        inclinations_deg=(40.0, 70.0),
        eddington_ratios=(0.06, 0.18),
        f_col_true_values=(1.7, 1.9),
        f_col_fit=1.7,
        inner_stress_delta_eta=(0.0, 0.02),
        replicate_count=20,
        energy_min_kev=0.1,
        energy_max_kev=20.0,
        energy_bin_count=32,
        spin_grid_count=401,
        gaussian_relative_error=0.03,
        thresholds=ScreeningThresholds(),
    )


def load_screening_config(path: Path) -> ScreeningConfig:
    """Load a Phase 11 screening configuration from YAML."""

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        msg = "screening config must be a mapping"
        raise TypeError(msg)
    return screening_config_from_mapping(raw)


def screening_config_from_mapping(raw: Mapping[str, Any]) -> ScreeningConfig:
    """Build a screening config from a parsed mapping."""

    default = default_screening_config()
    thresholds_raw = raw.get("thresholds", {})
    if thresholds_raw is None:
        thresholds_raw = {}
    if not isinstance(thresholds_raw, Mapping):
        msg = "thresholds must be a mapping"
        raise TypeError(msg)
    thresholds = ScreeningThresholds(
        high_bias_abs=_float_value(
            thresholds_raw,
            "high_bias_abs",
            default.thresholds.high_bias_abs,
        ),
        low_identifiability_width_68=_float_value(
            thresholds_raw,
            "low_identifiability_width_68",
            default.thresholds.low_identifiability_width_68,
        ),
        chi2_per_dof_max=_float_value(
            thresholds_raw,
            "chi2_per_dof_max",
            default.thresholds.chi2_per_dof_max,
        ),
        failure_rate_max=_float_value(
            thresholds_raw,
            "failure_rate_max",
            default.thresholds.failure_rate_max,
        ),
    )
    config = ScreeningConfig(
        config_version=str(raw.get("config_version", default.config_version)),
        master_seed=_int_value(raw, "master_seed", default.master_seed),
        spins=_float_tuple(raw, "spins", default.spins),
        inclinations_deg=_float_tuple(
            raw,
            "inclinations_deg",
            default.inclinations_deg,
        ),
        eddington_ratios=_float_tuple(
            raw,
            "eddington_ratios",
            default.eddington_ratios,
        ),
        f_col_true_values=_float_tuple(
            raw,
            "f_col_true_values",
            default.f_col_true_values,
        ),
        f_col_fit=_float_value(raw, "f_col_fit", default.f_col_fit),
        inner_stress_delta_eta=_float_tuple(
            raw,
            "inner_stress_delta_eta",
            default.inner_stress_delta_eta,
        ),
        replicate_count=_int_value(raw, "replicate_count", default.replicate_count),
        energy_min_kev=_float_value(raw, "energy_min_kev", default.energy_min_kev),
        energy_max_kev=_float_value(raw, "energy_max_kev", default.energy_max_kev),
        energy_bin_count=_int_value(
            raw,
            "energy_bin_count",
            default.energy_bin_count,
        ),
        spin_grid_count=_int_value(raw, "spin_grid_count", default.spin_grid_count),
        gaussian_relative_error=_float_value(
            raw,
            "gaussian_relative_error",
            default.gaussian_relative_error,
        ),
        thresholds=thresholds,
    )
    validate_screening_config(config)
    return config


def _float_value(raw: Mapping[str, Any], key: str, default: float) -> float:
    return float(raw.get(key, default))


def _int_value(raw: Mapping[str, Any], key: str, default: int) -> int:
    return int(raw.get(key, default))


def _float_tuple(
    raw: Mapping[str, Any],
    key: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    value = raw.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"{key} must be a sequence"
        raise TypeError(msg)
    return tuple(float(item) for item in value)


def validate_screening_config(config: ScreeningConfig) -> None:
    """Validate a screening configuration."""

    if config.master_seed < 0:
        msg = "master_seed must be nonnegative"
        raise ValueError(msg)
    if config.replicate_count < 1:
        msg = "replicate_count must be positive"
        raise ValueError(msg)
    if config.energy_bin_count < 2:
        msg = "energy_bin_count must be at least two"
        raise ValueError(msg)
    if config.spin_grid_count < 11:
        msg = "spin_grid_count must be at least eleven"
        raise ValueError(msg)
    if config.energy_min_kev <= 0.0 or config.energy_max_kev <= config.energy_min_kev:
        msg = "energy bounds must be positive and ordered"
        raise ValueError(msg)
    if config.gaussian_relative_error <= 0.0:
        msg = "gaussian_relative_error must be positive"
        raise ValueError(msg)
    for spin in config.spins:
        if not -0.998 <= spin <= 0.998:
            msg = "spins must lie within the frozen Phase 10 spin prior"
            raise ValueError(msg)
    for delta_eta in config.inner_stress_delta_eta:
        if not 0.0 <= delta_eta <= 0.1:
            msg = "inner_stress_delta_eta must lie in the controlled Phase 8 domain"
            raise ValueError(msg)
    for value_name, values in (
        ("inclinations_deg", config.inclinations_deg),
        ("eddington_ratios", config.eddington_ratios),
        ("f_col_true_values", config.f_col_true_values),
    ):
        if not values or any(not isfinite(value) or value <= 0.0 for value in values):
            msg = f"{value_name} must contain positive finite values"
            raise ValueError(msg)
    thresholds = config.thresholds
    if thresholds.high_bias_abs <= 0.0:
        msg = "high_bias_abs must be positive"
        raise ValueError(msg)
    if thresholds.low_identifiability_width_68 <= 0.0:
        msg = "low_identifiability_width_68 must be positive"
        raise ValueError(msg)
    if thresholds.chi2_per_dof_max <= 0.0:
        msg = "chi2_per_dof_max must be positive"
        raise ValueError(msg)
    if thresholds.failure_rate_max < 0.0:
        msg = "failure_rate_max must be nonnegative"
        raise ValueError(msg)


def build_screening_conditions(
    config: ScreeningConfig,
) -> tuple[ScreeningCondition, ...]:
    """Return deterministic coarse-grid screening conditions."""

    validate_screening_config(config)
    conditions: list[ScreeningCondition] = []
    for spin in config.spins:
        for inclination in config.inclinations_deg:
            for eddington_ratio in config.eddington_ratios:
                for f_col_true in config.f_col_true_values:
                    for delta_eta in config.inner_stress_delta_eta:
                        condition_id = (
                            f"a{spin:+.3f}_i{inclination:04.1f}_"
                            f"l{eddington_ratio:.3f}_fc{f_col_true:.2f}_"
                            f"de{delta_eta:.3f}"
                        )
                        conditions.append(
                            ScreeningCondition(
                                condition_id=_sanitize_condition_id(condition_id),
                                spin_true=spin,
                                inclination_deg=inclination,
                                eddington_ratio=eddington_ratio,
                                f_col_true=f_col_true,
                                f_col_fit=config.f_col_fit,
                                inner_stress_delta_eta=delta_eta,
                                replicate_count=config.replicate_count,
                            )
                        )
    return tuple(conditions)


def _sanitize_condition_id(condition_id: str) -> str:
    return (
        condition_id.replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace("_", "-")
    )


def run_screening_campaign(config: ScreeningConfig) -> ScreeningCampaignResult:
    """Run the deterministic Phase 11 coarse screening campaign."""

    conditions = build_screening_conditions(config)
    bins = make_log_energy_bins(
        energy_min_kev=config.energy_min_kev,
        energy_max_kev=config.energy_max_kev,
        bin_count=config.energy_bin_count,
    )
    spin_grid = np.linspace(-0.998, 0.998, config.spin_grid_count, dtype=np.float64)
    replicates: list[ReplicateResult] = []
    summaries: list[ConditionSummary] = []
    for condition in conditions:
        condition_replicates: list[ReplicateResult] = []
        for replicate_index in range(condition.replicate_count):
            result = _run_screening_replicate(
                config,
                condition,
                bins,
                spin_grid,
                replicate_index,
            )
            condition_replicates.append(result)
            replicates.append(result)
        summaries.append(
            summarize_condition(
                condition,
                condition_replicates,
                config.thresholds,
            )
        )
    confirmatory_conditions = tuple(
        summary for summary in summaries if summary.needs_refinement
    )
    return ScreeningCampaignResult(
        config=config,
        conditions=conditions,
        replicates=tuple(replicates),
        summaries=tuple(summaries),
        confirmatory_conditions=confirmatory_conditions,
    )


def _run_screening_replicate(
    config: ScreeningConfig,
    condition: ScreeningCondition,
    bins: EnergyBins,
    spin_grid: FloatArray,
    replicate_index: int,
) -> ReplicateResult:
    started = perf_counter()
    noise_seed = derive_seed(
        config.master_seed,
        condition.condition_id,
        replicate_index,
        "screening",
    )
    try:
        photon_flux = screening_proxy_photon_flux(
            bins,
            condition,
            condition.spin_true,
        )
        epoch = generate_gaussian_epoch(
            bins,
            photon_flux,
            GaussianRelativeNoise(config.gaussian_relative_error),
            master_seed=noise_seed,
            condition_label=condition.condition_id,
            epoch_index=replicate_index,
            truth_metadata={
                "spin_true": condition.spin_true,
                "inclination_deg": condition.inclination_deg,
                "eddington_ratio": condition.eddington_ratio,
                "f_col_true": condition.f_col_true,
                "f_col_fit": condition.f_col_fit,
                "inner_stress_delta_eta": condition.inner_stress_delta_eta,
            },
        )
        fit = _fit_spin_grid(bins, condition, epoch, spin_grid)
        runtime = perf_counter() - started
        return ReplicateResult(
            condition_id=condition.condition_id,
            replicate_index=replicate_index,
            status="COMPLETED",
            failure_cause="",
            spin_true=condition.spin_true,
            spin_map=fit["spin_map"],
            spin_mean=fit["spin_mean"],
            bias=fit["spin_mean"] - condition.spin_true,
            ci68_lower=fit["ci68_lower"],
            ci68_upper=fit["ci68_upper"],
            ci95_lower=fit["ci95_lower"],
            ci95_upper=fit["ci95_upper"],
            chi2_per_dof=fit["chi2_per_dof"],
            runtime_s=runtime,
            noise_seed=noise_seed,
        )
    except (FloatingPointError, ValueError) as exc:
        runtime = perf_counter() - started
        return ReplicateResult(
            condition_id=condition.condition_id,
            replicate_index=replicate_index,
            status="FAILED",
            failure_cause=str(exc),
            spin_true=condition.spin_true,
            spin_map=float("nan"),
            spin_mean=float("nan"),
            bias=float("nan"),
            ci68_lower=float("nan"),
            ci68_upper=float("nan"),
            ci95_lower=float("nan"),
            ci95_upper=float("nan"),
            chi2_per_dof=float("nan"),
            runtime_s=runtime,
            noise_seed=noise_seed,
        )


def screening_proxy_photon_flux(
    bins: EnergyBins,
    condition: ScreeningCondition,
    spin: float,
) -> FloatArray:
    """Return the Phase 11/12 detector-independent proxy photon flux."""

    template = np.exp(-bins.centers_kev / 5.0) * np.sqrt(condition.eddington_ratio)
    inclination_factor = 1.0 + (0.001 * (condition.inclination_deg - 45.0))
    spin_factor = 1.0 + (0.25 * spin)
    f_col_misspecification = 0.35 * (condition.f_col_true - condition.f_col_fit)
    stress_misspecification = 2.0 * condition.inner_stress_delta_eta
    amplitude = inclination_factor * (
        spin_factor + f_col_misspecification + stress_misspecification
    )
    if amplitude <= 0.0:
        msg = "proxy screening amplitude is nonpositive"
        raise ValueError(msg)
    photon_flux: FloatArray = np.asarray(template * amplitude * bins.widths_kev)
    return photon_flux


def _fit_spin_grid(
    bins: EnergyBins,
    condition: ScreeningCondition,
    epoch: SyntheticEpoch,
    spin_grid: FloatArray,
) -> dict[str, float]:
    log_posterior = np.empty_like(spin_grid)
    for index, spin in enumerate(spin_grid):
        model_photon_flux = screening_proxy_photon_flux(
            bins,
            ScreeningCondition(
                condition_id=condition.condition_id,
                spin_true=condition.spin_true,
                inclination_deg=condition.inclination_deg,
                eddington_ratio=condition.eddington_ratio,
                f_col_true=condition.f_col_fit,
                f_col_fit=condition.f_col_fit,
                inner_stress_delta_eta=0.0,
                replicate_count=condition.replicate_count,
            ),
            float(spin),
        )
        model_energy_flux = bins.centers_kev * model_photon_flux
        log_posterior[index] = gaussian_log_likelihood(
            epoch.observed,
            model_energy_flux,
            epoch.variance,
        )
    if not np.any(np.isfinite(log_posterior)):
        msg = "no finite spin-grid likelihood values"
        raise FloatingPointError(msg)
    weights = _normalized_weights(log_posterior)
    spin_mean = float(np.sum(weights * spin_grid))
    spin_map = float(spin_grid[int(np.argmax(log_posterior))])
    ci68 = _weighted_interval(spin_grid, weights, 0.68)
    ci95 = _weighted_interval(spin_grid, weights, 0.95)
    best_photon_flux = screening_proxy_photon_flux(
        bins,
        ScreeningCondition(
            condition_id=condition.condition_id,
            spin_true=condition.spin_true,
            inclination_deg=condition.inclination_deg,
            eddington_ratio=condition.eddington_ratio,
            f_col_true=condition.f_col_fit,
            f_col_fit=condition.f_col_fit,
            inner_stress_delta_eta=0.0,
            replicate_count=condition.replicate_count,
        ),
        spin_map,
    )
    chi2_per_dof = _chi2_per_dof(
        epoch.observed,
        bins.centers_kev * best_photon_flux,
        epoch.variance,
        fit_parameter_count=1,
    )
    return {
        "spin_map": spin_map,
        "spin_mean": spin_mean,
        "ci68_lower": ci68[0],
        "ci68_upper": ci68[1],
        "ci95_lower": ci95[0],
        "ci95_upper": ci95[1],
        "chi2_per_dof": chi2_per_dof,
    }


def _normalized_weights(log_values: FloatArray) -> FloatArray:
    finite = np.isfinite(log_values)
    shifted = np.full_like(log_values, -np.inf)
    shifted[finite] = log_values[finite] - float(np.max(log_values[finite]))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0:
        msg = "posterior weights underflowed"
        raise FloatingPointError(msg)
    return weights / total


def _weighted_interval(
    grid: FloatArray,
    weights: FloatArray,
    level: float,
) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    lower = float(np.interp(alpha, cumulative, grid))
    upper = float(np.interp(1.0 - alpha, cumulative, grid))
    return lower, upper


def _chi2_per_dof(
    observed: ArrayLike,
    model: ArrayLike,
    variance: ArrayLike,
    *,
    fit_parameter_count: int,
) -> float:
    observed_array = np.asarray(observed, dtype=np.float64)
    model_array = np.asarray(model, dtype=np.float64)
    variance_array = np.asarray(variance, dtype=np.float64)
    dof = observed_array.size - fit_parameter_count
    if dof <= 0:
        msg = "chi2_per_dof requires more bins than fitted parameters"
        raise ValueError(msg)
    chi2 = np.sum(np.square(observed_array - model_array) / variance_array)
    return float(chi2 / dof)


def summarize_condition(
    condition: ScreeningCondition,
    replicates: Sequence[ReplicateResult],
    thresholds: ScreeningThresholds,
) -> ConditionSummary:
    """Summarize one condition without silently dropping failures."""

    if len(replicates) != condition.replicate_count:
        msg = "replicate count does not match condition plan"
        raise ValueError(msg)
    completed = [
        replicate for replicate in replicates if replicate.status == "COMPLETED"
    ]
    failed = [replicate for replicate in replicates if replicate.status == "FAILED"]
    failure_rate = len(failed) / condition.replicate_count
    if completed:
        biases = np.array([replicate.bias for replicate in completed], dtype=np.float64)
        widths68 = np.array(
            [replicate.ci68_upper - replicate.ci68_lower for replicate in completed],
            dtype=np.float64,
        )
        widths95 = np.array(
            [replicate.ci95_upper - replicate.ci95_lower for replicate in completed],
            dtype=np.float64,
        )
        chi2 = np.array(
            [replicate.chi2_per_dof for replicate in completed],
            dtype=np.float64,
        )
        runtime = np.array([replicate.runtime_s for replicate in completed])
        coverage68 = np.mean(
            [
                replicate.ci68_lower <= condition.spin_true <= replicate.ci68_upper
                for replicate in completed
            ]
        )
        coverage95 = np.mean(
            [
                replicate.ci95_lower <= condition.spin_true <= replicate.ci95_upper
                for replicate in completed
            ]
        )
        mean_bias = float(np.mean(biases))
        median_bias = float(np.median(biases))
        rmse = float(sqrt(float(np.mean(np.square(biases)))))
        mean_width_68 = float(np.mean(widths68))
        mean_width_95 = float(np.mean(widths95))
        mean_chi2 = float(np.mean(chi2))
        mean_runtime = float(np.mean(runtime))
        coverage_68_value = float(coverage68)
        coverage_95_value = float(coverage95)
    else:
        mean_bias = float("nan")
        median_bias = float("nan")
        rmse = float("nan")
        mean_width_68 = float("nan")
        mean_width_95 = float("nan")
        mean_chi2 = float("nan")
        mean_runtime = float("nan")
        coverage_68_value = float("nan")
        coverage_95_value = float("nan")

    high_bias = bool(isfinite(mean_bias) and abs(mean_bias) >= thresholds.high_bias_abs)
    low_identifiability = bool(
        isfinite(mean_width_68)
        and mean_width_68 >= thresholds.low_identifiability_width_68
    )
    poor_fit = bool(isfinite(mean_chi2) and mean_chi2 >= thresholds.chi2_per_dof_max)
    needs_refinement = (
        high_bias
        or low_identifiability
        or poor_fit
        or failure_rate > thresholds.failure_rate_max
    )
    status: ConditionStatus = "COMPLETED" if not failed else "FAILED"
    failure_cause = "; ".join(sorted({replicate.failure_cause for replicate in failed}))
    return ConditionSummary(
        condition_id=condition.condition_id,
        status=status,
        failure_cause=failure_cause,
        planned_replicates=condition.replicate_count,
        completed_replicates=len(completed),
        failed_replicates=len(failed),
        failure_rate=failure_rate,
        spin_true=condition.spin_true,
        inclination_deg=condition.inclination_deg,
        eddington_ratio=condition.eddington_ratio,
        f_col_true=condition.f_col_true,
        f_col_fit=condition.f_col_fit,
        inner_stress_delta_eta=condition.inner_stress_delta_eta,
        mean_bias=mean_bias,
        median_bias=median_bias,
        rmse=rmse,
        mean_width_68=mean_width_68,
        mean_width_95=mean_width_95,
        coverage_68=coverage_68_value,
        coverage_95=coverage_95_value,
        mean_chi2_per_dof=mean_chi2,
        mean_runtime_s=mean_runtime,
        high_bias_flag=high_bias,
        low_identifiability_flag=low_identifiability,
        poor_fit_flag=poor_fit,
        needs_refinement=needs_refinement,
    )


def write_screening_outputs(
    result: ScreeningCampaignResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write Phase 11 machine-readable screening outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = output_dir / "phase11_conditions.csv"
    status_path = output_dir / "phase11_condition_status.csv"
    replicates_path = output_dir / "phase11_replicates.csv"
    confirmatory_path = output_dir / "phase11_confirmatory_conditions.csv"
    protocol_path = output_dir / "phase11_confirmatory_protocol.md"
    _write_csv(
        conditions_path,
        [_condition_row(condition) for condition in result.conditions],
    )
    _write_csv(status_path, [_summary_row(summary) for summary in result.summaries])
    _write_csv(
        replicates_path,
        [_replicate_row(replicate) for replicate in result.replicates],
    )
    _write_csv(
        confirmatory_path,
        [_summary_row(summary) for summary in result.confirmatory_conditions],
    )
    protocol_path.write_text(_confirmatory_protocol(result), encoding="utf-8")
    return {
        "conditions": conditions_path,
        "status": status_path,
        "replicates": replicates_path,
        "confirmatory_conditions": confirmatory_path,
        "confirmatory_protocol": protocol_path,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _condition_row(condition: ScreeningCondition) -> dict[str, object]:
    return {
        "condition_id": condition.condition_id,
        "spin_true": condition.spin_true,
        "inclination_deg": condition.inclination_deg,
        "eddington_ratio": condition.eddington_ratio,
        "f_col_true": condition.f_col_true,
        "f_col_fit": condition.f_col_fit,
        "inner_stress_delta_eta": condition.inner_stress_delta_eta,
        "replicate_count": condition.replicate_count,
    }


def _replicate_row(replicate: ReplicateResult) -> dict[str, object]:
    return {
        "condition_id": replicate.condition_id,
        "replicate_index": replicate.replicate_index,
        "status": replicate.status,
        "failure_cause": replicate.failure_cause,
        "spin_true": replicate.spin_true,
        "spin_map": replicate.spin_map,
        "spin_mean": replicate.spin_mean,
        "bias": replicate.bias,
        "ci68_lower": replicate.ci68_lower,
        "ci68_upper": replicate.ci68_upper,
        "ci95_lower": replicate.ci95_lower,
        "ci95_upper": replicate.ci95_upper,
        "chi2_per_dof": replicate.chi2_per_dof,
        "runtime_s": replicate.runtime_s,
        "noise_seed": replicate.noise_seed,
    }


def _summary_row(summary: ConditionSummary) -> dict[str, object]:
    return {
        "condition_id": summary.condition_id,
        "status": summary.status,
        "failure_cause": summary.failure_cause,
        "planned_replicates": summary.planned_replicates,
        "completed_replicates": summary.completed_replicates,
        "failed_replicates": summary.failed_replicates,
        "failure_rate": summary.failure_rate,
        "spin_true": summary.spin_true,
        "inclination_deg": summary.inclination_deg,
        "eddington_ratio": summary.eddington_ratio,
        "f_col_true": summary.f_col_true,
        "f_col_fit": summary.f_col_fit,
        "inner_stress_delta_eta": summary.inner_stress_delta_eta,
        "mean_bias": summary.mean_bias,
        "median_bias": summary.median_bias,
        "rmse": summary.rmse,
        "mean_width_68": summary.mean_width_68,
        "mean_width_95": summary.mean_width_95,
        "coverage_68": summary.coverage_68,
        "coverage_95": summary.coverage_95,
        "mean_chi2_per_dof": summary.mean_chi2_per_dof,
        "mean_runtime_s": summary.mean_runtime_s,
        "high_bias_flag": summary.high_bias_flag,
        "low_identifiability_flag": summary.low_identifiability_flag,
        "poor_fit_flag": summary.poor_fit_flag,
        "needs_refinement": summary.needs_refinement,
    }


def _confirmatory_protocol(result: ScreeningCampaignResult) -> str:
    thresholds = result.config.thresholds
    return "\n".join(
        [
            "# Phase 11 Frozen Confirmatory Protocol",
            "",
            "Status: frozen before Phase 12 confirmatory truths are generated.",
            "",
            "This protocol is based on Phase 11 screening artifacts only. It does",
            "not state final astrophysical conclusions.",
            "",
            "## Refinement Criteria",
            "",
            f"- High bias: `abs(mean_bias) >= {thresholds.high_bias_abs}`.",
            "- Low identifiability: "
            f"`mean_width_68 >= {thresholds.low_identifiability_width_68}`.",
            f"- Poor fit: `mean_chi2_per_dof >= {thresholds.chi2_per_dof_max}`.",
            f"- Failure accounting: `failure_rate > {thresholds.failure_rate_max}`.",
            "",
            "Any condition meeting at least one criterion is included in",
            "`phase11_confirmatory_conditions.csv`.",
            "",
            "## Accounting Rules",
            "",
            "- Failed replicates are retained in `phase11_replicates.csv`.",
            "- Conditions are never removed post hoc without a status row and",
            "  failure cause.",
            "- Phase 12 must use this locked condition list unless a new protocol",
            "  version is created before hidden truths are generated.",
            "",
            "## Counts",
            "",
            f"- Screening conditions: {len(result.conditions)}.",
            "- Conditions selected for refinement: "
            f"{len(result.confirmatory_conditions)}.",
            f"- Planned replicates per condition: {result.config.replicate_count}.",
            "",
        ]
    )
