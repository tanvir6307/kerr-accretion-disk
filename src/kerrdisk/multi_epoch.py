"""Joint multi-epoch comparison utilities for pre-manuscript outputs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import blake2b
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from kerrdisk.confirmatory import (
    ConfirmatoryCondition,
    ConfirmatoryConfig,
    TransferCache,
    _energy_flux_for_spin,
    _fit_spin_grid_vectorized,
    _model_energy_flux_matrix,
    _transfer_map_for,
    load_confirmatory_config,
    load_locked_conditions,
)
from kerrdisk.synthetic import EnergyBins, derive_seed, make_log_energy_bins

type FloatArray = NDArray[np.float64]
type MultiEpochStatus = Literal["COMPLETED", "FAILED"]


@dataclass(frozen=True)
class MultiEpochConfig:
    """Configuration for the two-epoch shared-spin comparison."""

    config_version: str
    confirmatory_config_path: Path
    master_seed: int
    replicate_count: int
    min_epoch_count: int
    width_reduction_required_fraction: float


@dataclass(frozen=True)
class MultiEpochGroup:
    """Conditions that share fitted global parameters across luminosity epochs."""

    group_id: str
    spin_true: float
    inclination_deg: float
    f_col_true: float
    f_col_fit: float
    inner_stress_delta_eta: float
    conditions: tuple[ConfirmatoryCondition, ...]


@dataclass(frozen=True)
class MultiEpochReplicate:
    """One separate-versus-joint multi-epoch replicate."""

    group_id: str
    replicate_index: int
    status: MultiEpochStatus
    failure_cause: str
    epoch_count: int
    spin_true: float
    single_epoch_mean_spin: float
    joint_spin_mean: float
    single_epoch_mean_bias: float
    joint_bias: float
    single_epoch_mean_abs_bias: float
    joint_abs_bias: float
    single_epoch_mean_width_68: float
    joint_width_68: float
    single_epoch_mean_width_95: float
    joint_width_95: float
    single_epoch_coverage_68: float
    joint_coverage_68: float
    single_epoch_coverage_95: float
    joint_coverage_95: float
    joint_width_reduction_fraction: float
    runtime_s: float
    noise_seed: int


@dataclass(frozen=True)
class MultiEpochSummary:
    """Group-level summary of separate-versus-joint fitting."""

    group_id: str
    status: MultiEpochStatus
    failure_cause: str
    spin_true: float
    inclination_deg: float
    f_col_true: float
    f_col_fit: float
    inner_stress_delta_eta: float
    epoch_count: int
    eddington_ratios: str
    planned_replicates: int
    completed_replicates: int
    failed_replicates: int
    failure_rate: float
    mean_single_epoch_bias: float
    mean_joint_bias: float
    mean_single_epoch_abs_bias: float
    mean_joint_abs_bias: float
    single_epoch_rmse: float
    joint_rmse: float
    mean_single_epoch_width_68: float
    mean_joint_width_68: float
    mean_single_epoch_width_95: float
    mean_joint_width_95: float
    coverage_68_single_epoch: float
    coverage_68_joint: float
    coverage_95_single_epoch: float
    coverage_95_joint: float
    width_68_reduction_fraction: float
    width_95_reduction_fraction: float
    joint_abs_bias_minus_single_epoch_abs_bias: float


@dataclass(frozen=True)
class MultiEpochCampaignResult:
    """Full multi-epoch comparison result."""

    config: MultiEpochConfig
    confirmatory_config: ConfirmatoryConfig
    groups: tuple[MultiEpochGroup, ...]
    replicates: tuple[MultiEpochReplicate, ...]
    summaries: tuple[MultiEpochSummary, ...]


def default_multi_epoch_config() -> MultiEpochConfig:
    """Return the default pre-Phase-14 multi-epoch comparison config."""

    return MultiEpochConfig(
        config_version="phase13p5_multi_epoch_v1",
        confirmatory_config_path=Path("configs/production/phase12_confirmatory.yaml"),
        master_seed=20260621,
        replicate_count=100,
        min_epoch_count=2,
        width_reduction_required_fraction=0.0,
    )


def load_multi_epoch_config(path: Path) -> MultiEpochConfig:
    """Load multi-epoch config from YAML."""

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        msg = "multi-epoch config must be a mapping"
        raise TypeError(msg)
    return multi_epoch_config_from_mapping(raw)


def multi_epoch_config_from_mapping(raw: Mapping[str, Any]) -> MultiEpochConfig:
    """Build a multi-epoch config from a mapping."""

    default = default_multi_epoch_config()
    config = MultiEpochConfig(
        config_version=str(raw.get("config_version", default.config_version)),
        confirmatory_config_path=Path(
            str(raw.get("confirmatory_config_path", default.confirmatory_config_path))
        ),
        master_seed=int(raw.get("master_seed", default.master_seed)),
        replicate_count=int(raw.get("replicate_count", default.replicate_count)),
        min_epoch_count=int(raw.get("min_epoch_count", default.min_epoch_count)),
        width_reduction_required_fraction=float(
            raw.get(
                "width_reduction_required_fraction",
                default.width_reduction_required_fraction,
            )
        ),
    )
    validate_multi_epoch_config(config)
    return config


def validate_multi_epoch_config(config: MultiEpochConfig) -> None:
    """Validate multi-epoch config values."""

    if config.master_seed < 0:
        msg = "master_seed must be nonnegative"
        raise ValueError(msg)
    if config.replicate_count < 1:
        msg = "replicate_count must be positive"
        raise ValueError(msg)
    if config.min_epoch_count < 2:
        msg = "min_epoch_count must be at least two"
        raise ValueError(msg)
    if (
        not isfinite(config.width_reduction_required_fraction)
        or config.width_reduction_required_fraction < 0.0
        or config.width_reduction_required_fraction >= 1.0
    ):
        msg = "width_reduction_required_fraction must satisfy 0 <= value < 1"
        raise ValueError(msg)


def build_multi_epoch_groups(
    conditions: Sequence[ConfirmatoryCondition],
    *,
    min_epoch_count: int = 2,
) -> tuple[MultiEpochGroup, ...]:
    """Group locked conditions into shared-spin multi-epoch datasets."""

    grouped: dict[tuple[float, float, float, float, float], list[ConfirmatoryCondition]]
    grouped = {}
    for condition in conditions:
        key = (
            condition.spin_true,
            condition.inclination_deg,
            condition.f_col_true,
            condition.f_col_fit,
            condition.inner_stress_delta_eta,
        )
        grouped.setdefault(key, []).append(condition)

    result: list[MultiEpochGroup] = []
    for key, items in sorted(grouped.items()):
        unique_luminosities = {condition.eddington_ratio for condition in items}
        if len(items) < min_epoch_count or len(unique_luminosities) < min_epoch_count:
            continue
        selected = tuple(
            sorted(items, key=lambda condition: condition.eddington_ratio)[
                :min_epoch_count
            ]
        )
        spin, inclination, f_col_true, f_col_fit, delta_eta = key
        result.append(
            MultiEpochGroup(
                group_id=_group_id(selected),
                spin_true=spin,
                inclination_deg=inclination,
                f_col_true=f_col_true,
                f_col_fit=f_col_fit,
                inner_stress_delta_eta=delta_eta,
                conditions=selected,
            )
        )
    if not result:
        msg = "no multi-epoch groups could be built from locked conditions"
        raise ValueError(msg)
    return tuple(result)


def run_multi_epoch_campaign(config: MultiEpochConfig) -> MultiEpochCampaignResult:
    """Run the separate-versus-joint shared-spin comparison."""

    validate_multi_epoch_config(config)
    confirmatory_config = load_confirmatory_config(config.confirmatory_config_path)
    conditions = load_locked_conditions(
        confirmatory_config.locked_conditions_path,
        master_seed=confirmatory_config.master_seed,
    )
    groups = build_multi_epoch_groups(
        conditions,
        min_epoch_count=config.min_epoch_count,
    )
    run_config = replace(confirmatory_config, replicate_count=config.replicate_count)
    bins = make_log_energy_bins(
        energy_min_kev=run_config.energy_min_kev,
        energy_max_kev=run_config.energy_max_kev,
        bin_count=run_config.energy_bin_count,
    )
    spin_grid = np.linspace(-0.998, 0.998, run_config.spin_grid_count)
    transfer_cache: TransferCache = {}

    replicates: list[MultiEpochReplicate] = []
    summaries: list[MultiEpochSummary] = []
    for group in groups:
        model_matrices = tuple(
            _model_energy_flux_matrix(
                run_config,
                bins,
                condition,
                spin_grid,
                transfer_cache,
            )
            for condition in group.conditions
        )
        true_fluxes = tuple(
            _true_energy_flux_for_condition(run_config, bins, condition, transfer_cache)
            for condition in group.conditions
        )
        group_replicates = tuple(
            _run_multi_epoch_replicate(
                config,
                run_config,
                group,
                spin_grid,
                model_matrices,
                true_fluxes,
                replicate_index,
            )
            for replicate_index in range(config.replicate_count)
        )
        replicates.extend(group_replicates)
        summaries.append(summarize_multi_epoch_group(group, group_replicates))

    return MultiEpochCampaignResult(
        config=config,
        confirmatory_config=run_config,
        groups=groups,
        replicates=tuple(replicates),
        summaries=tuple(summaries),
    )


def summarize_multi_epoch_group(
    group: MultiEpochGroup,
    replicates: Sequence[MultiEpochReplicate],
) -> MultiEpochSummary:
    """Summarize one multi-epoch group."""

    completed = [
        replicate for replicate in replicates if replicate.status == "COMPLETED"
    ]
    failed = [replicate for replicate in replicates if replicate.status == "FAILED"]
    if completed:
        single_bias = np.array(
            [replicate.single_epoch_mean_bias for replicate in completed]
        )
        joint_bias = np.array([replicate.joint_bias for replicate in completed])
        single_abs = np.array(
            [replicate.single_epoch_mean_abs_bias for replicate in completed]
        )
        joint_abs = np.array([replicate.joint_abs_bias for replicate in completed])
        single_w68 = np.array(
            [replicate.single_epoch_mean_width_68 for replicate in completed]
        )
        joint_w68 = np.array([replicate.joint_width_68 for replicate in completed])
        single_w95 = np.array(
            [replicate.single_epoch_mean_width_95 for replicate in completed]
        )
        joint_w95 = np.array([replicate.joint_width_95 for replicate in completed])
        mean_single_bias = float(np.mean(single_bias))
        mean_joint_bias = float(np.mean(joint_bias))
        mean_single_abs = float(np.mean(single_abs))
        mean_joint_abs = float(np.mean(joint_abs))
        single_rmse = float(np.sqrt(np.mean(np.square(single_bias))))
        joint_rmse = float(np.sqrt(np.mean(np.square(joint_bias))))
        mean_single_w68 = float(np.mean(single_w68))
        mean_joint_w68 = float(np.mean(joint_w68))
        mean_single_w95 = float(np.mean(single_w95))
        mean_joint_w95 = float(np.mean(joint_w95))
        cov68_single = float(
            np.mean([replicate.single_epoch_coverage_68 for replicate in completed])
        )
        cov68_joint = float(
            np.mean([replicate.joint_coverage_68 for replicate in completed])
        )
        cov95_single = float(
            np.mean([replicate.single_epoch_coverage_95 for replicate in completed])
        )
        cov95_joint = float(
            np.mean([replicate.joint_coverage_95 for replicate in completed])
        )
    else:
        mean_single_bias = mean_joint_bias = float("nan")
        mean_single_abs = mean_joint_abs = float("nan")
        single_rmse = joint_rmse = float("nan")
        mean_single_w68 = mean_joint_w68 = float("nan")
        mean_single_w95 = mean_joint_w95 = float("nan")
        cov68_single = cov68_joint = float("nan")
        cov95_single = cov95_joint = float("nan")

    return MultiEpochSummary(
        group_id=group.group_id,
        status="COMPLETED" if not failed else "FAILED",
        failure_cause="; ".join(
            sorted({replicate.failure_cause for replicate in failed})
        ),
        spin_true=group.spin_true,
        inclination_deg=group.inclination_deg,
        f_col_true=group.f_col_true,
        f_col_fit=group.f_col_fit,
        inner_stress_delta_eta=group.inner_stress_delta_eta,
        epoch_count=len(group.conditions),
        eddington_ratios=";".join(
            f"{condition.eddington_ratio:.6g}" for condition in group.conditions
        ),
        planned_replicates=len(replicates),
        completed_replicates=len(completed),
        failed_replicates=len(failed),
        failure_rate=len(failed) / len(replicates) if replicates else 1.0,
        mean_single_epoch_bias=mean_single_bias,
        mean_joint_bias=mean_joint_bias,
        mean_single_epoch_abs_bias=mean_single_abs,
        mean_joint_abs_bias=mean_joint_abs,
        single_epoch_rmse=single_rmse,
        joint_rmse=joint_rmse,
        mean_single_epoch_width_68=mean_single_w68,
        mean_joint_width_68=mean_joint_w68,
        mean_single_epoch_width_95=mean_single_w95,
        mean_joint_width_95=mean_joint_w95,
        coverage_68_single_epoch=cov68_single,
        coverage_68_joint=cov68_joint,
        coverage_95_single_epoch=cov95_single,
        coverage_95_joint=cov95_joint,
        width_68_reduction_fraction=_fractional_reduction(
            mean_single_w68, mean_joint_w68
        ),
        width_95_reduction_fraction=_fractional_reduction(
            mean_single_w95, mean_joint_w95
        ),
        joint_abs_bias_minus_single_epoch_abs_bias=mean_joint_abs - mean_single_abs,
    )


def write_multi_epoch_outputs(
    result: MultiEpochCampaignResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write multi-epoch replicate and summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "groups": output_dir / "phase13p5_multi_epoch_groups.csv",
        "replicates": output_dir / "phase13p5_multi_epoch_replicates.csv",
        "summary": output_dir / "phase13p5_multi_epoch_summary.csv",
        "failure_summary": output_dir / "phase13p5_multi_epoch_failure_summary.csv",
        "analysis_freeze": output_dir / "phase13p5_multi_epoch_analysis_freeze.md",
    }
    _write_csv(paths["groups"], [_group_row(group) for group in result.groups])
    _write_csv(
        paths["replicates"],
        [_replicate_row(replicate) for replicate in result.replicates],
    )
    _write_csv(
        paths["summary"],
        [_summary_row(summary) for summary in result.summaries],
    )
    _write_csv(paths["failure_summary"], _failure_rows(result.summaries))
    paths["analysis_freeze"].write_text(
        _analysis_freeze_text(result),
        encoding="utf-8",
    )
    return paths


def _run_multi_epoch_replicate(
    config: MultiEpochConfig,
    run_config: ConfirmatoryConfig,
    group: MultiEpochGroup,
    spin_grid: FloatArray,
    model_matrices: Sequence[FloatArray],
    true_fluxes: Sequence[FloatArray],
    replicate_index: int,
) -> MultiEpochReplicate:
    started = perf_counter()
    noise_seed = derive_seed(
        config.master_seed,
        group.group_id,
        replicate_index,
        "multi_epoch",
    )
    try:
        observed: list[FloatArray] = []
        variances: list[FloatArray] = []
        for epoch_index, true_flux in enumerate(true_fluxes):
            epoch_seed = derive_seed(noise_seed, "epoch", epoch_index)
            rng = np.random.default_rng(epoch_seed)
            sigma = run_config.gaussian_relative_error * true_flux
            observed.append(rng.normal(loc=true_flux, scale=sigma))
            variances.append(sigma * sigma)

        single_fits = [
            _fit_spin_grid_vectorized(obs, var, matrix, spin_grid)
            for obs, var, matrix in zip(
                observed, variances, model_matrices, strict=True
            )
        ]
        joint_fit = _fit_joint_spin_grid(observed, variances, model_matrices, spin_grid)

        single_means = np.array([fit["spin_mean"] for fit in single_fits])
        single_biases = single_means - group.spin_true
        single_width68 = np.array(
            [fit["ci68_upper"] - fit["ci68_lower"] for fit in single_fits]
        )
        single_width95 = np.array(
            [fit["ci95_upper"] - fit["ci95_lower"] for fit in single_fits]
        )
        joint_width68 = joint_fit["ci68_upper"] - joint_fit["ci68_lower"]
        joint_width95 = joint_fit["ci95_upper"] - joint_fit["ci95_lower"]
        runtime = perf_counter() - started
        return MultiEpochReplicate(
            group_id=group.group_id,
            replicate_index=replicate_index,
            status="COMPLETED",
            failure_cause="",
            epoch_count=len(group.conditions),
            spin_true=group.spin_true,
            single_epoch_mean_spin=float(np.mean(single_means)),
            joint_spin_mean=joint_fit["spin_mean"],
            single_epoch_mean_bias=float(np.mean(single_biases)),
            joint_bias=joint_fit["spin_mean"] - group.spin_true,
            single_epoch_mean_abs_bias=float(np.mean(np.abs(single_biases))),
            joint_abs_bias=abs(joint_fit["spin_mean"] - group.spin_true),
            single_epoch_mean_width_68=float(np.mean(single_width68)),
            joint_width_68=joint_width68,
            single_epoch_mean_width_95=float(np.mean(single_width95)),
            joint_width_95=joint_width95,
            single_epoch_coverage_68=float(
                np.mean(
                    [
                        fit["ci68_lower"] <= group.spin_true <= fit["ci68_upper"]
                        for fit in single_fits
                    ]
                )
            ),
            joint_coverage_68=float(
                joint_fit["ci68_lower"] <= group.spin_true <= joint_fit["ci68_upper"]
            ),
            single_epoch_coverage_95=float(
                np.mean(
                    [
                        fit["ci95_lower"] <= group.spin_true <= fit["ci95_upper"]
                        for fit in single_fits
                    ]
                )
            ),
            joint_coverage_95=float(
                joint_fit["ci95_lower"] <= group.spin_true <= joint_fit["ci95_upper"]
            ),
            joint_width_reduction_fraction=_fractional_reduction(
                float(np.mean(single_width68)), joint_width68
            ),
            runtime_s=runtime,
            noise_seed=noise_seed,
        )
    except (FloatingPointError, ValueError) as exc:
        runtime = perf_counter() - started
        return MultiEpochReplicate(
            group_id=group.group_id,
            replicate_index=replicate_index,
            status="FAILED",
            failure_cause=str(exc),
            epoch_count=len(group.conditions),
            spin_true=group.spin_true,
            single_epoch_mean_spin=float("nan"),
            joint_spin_mean=float("nan"),
            single_epoch_mean_bias=float("nan"),
            joint_bias=float("nan"),
            single_epoch_mean_abs_bias=float("nan"),
            joint_abs_bias=float("nan"),
            single_epoch_mean_width_68=float("nan"),
            joint_width_68=float("nan"),
            single_epoch_mean_width_95=float("nan"),
            joint_width_95=float("nan"),
            single_epoch_coverage_68=float("nan"),
            joint_coverage_68=float("nan"),
            single_epoch_coverage_95=float("nan"),
            joint_coverage_95=float("nan"),
            joint_width_reduction_fraction=float("nan"),
            runtime_s=runtime,
            noise_seed=noise_seed,
        )


def _fit_joint_spin_grid(
    observed_by_epoch: Sequence[FloatArray],
    variance_by_epoch: Sequence[FloatArray],
    model_matrices: Sequence[FloatArray],
    spin_grid: FloatArray,
) -> dict[str, float]:
    log_likelihood = np.zeros(spin_grid.size, dtype=np.float64)
    for observed, variance, model_matrix in zip(
        observed_by_epoch,
        variance_by_epoch,
        model_matrices,
        strict=True,
    ):
        residual = observed[None, :] - model_matrix
        log_likelihood += -0.5 * np.sum(
            (residual * residual / variance[None, :])
            + np.log(2.0 * np.pi * variance)[None, :],
            axis=1,
        )
    if not np.any(np.isfinite(log_likelihood)):
        msg = "no finite joint spin-grid likelihood values"
        raise FloatingPointError(msg)
    weights = _normalized_weights(log_likelihood)
    ci68 = _weighted_interval(spin_grid, weights, 0.68)
    ci95 = _weighted_interval(spin_grid, weights, 0.95)
    map_index = int(np.argmax(log_likelihood))
    return {
        "spin_map": float(spin_grid[map_index]),
        "spin_mean": float(np.sum(weights * spin_grid)),
        "ci68_lower": ci68[0],
        "ci68_upper": ci68[1],
        "ci95_lower": ci95[0],
        "ci95_upper": ci95[1],
    }


def _true_energy_flux_for_condition(
    config: ConfirmatoryConfig,
    bins: EnergyBins,
    condition: ConfirmatoryCondition,
    transfer_cache: TransferCache,
) -> FloatArray:
    if config.model_backend == "ray_traced_transfer":
        _transfer_map_for(
            config,
            spin=condition.spin_true,
            inclination_deg=condition.inclination_deg,
            transfer_cache=transfer_cache,
        )
    return _energy_flux_for_spin(
        config,
        bins,
        condition,
        spin=condition.spin_true,
        fit_model=False,
        transfer_cache=transfer_cache,
    )


def _normalized_weights(log_values: FloatArray) -> FloatArray:
    shifted = log_values - float(np.max(log_values))
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
    return (
        float(np.interp(alpha, cumulative, grid)),
        float(np.interp(1.0 - alpha, cumulative, grid)),
    )


def _fractional_reduction(single_value: float, joint_value: float) -> float:
    if single_value <= 0.0 or not isfinite(single_value) or not isfinite(joint_value):
        return float("nan")
    return float(1.0 - (joint_value / single_value))


def _group_id(conditions: Sequence[ConfirmatoryCondition]) -> str:
    digest = blake2b(digest_size=6)
    for condition in conditions:
        digest.update(condition.condition_id.encode("utf-8"))
        digest.update(b"\0")
    return f"multi_{digest.hexdigest()}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _group_row(group: MultiEpochGroup) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "spin_true": group.spin_true,
        "inclination_deg": group.inclination_deg,
        "f_col_true": group.f_col_true,
        "f_col_fit": group.f_col_fit,
        "inner_stress_delta_eta": group.inner_stress_delta_eta,
        "epoch_count": len(group.conditions),
        "condition_ids": ";".join(
            condition.condition_id for condition in group.conditions
        ),
        "eddington_ratios": ";".join(
            f"{condition.eddington_ratio:.6g}" for condition in group.conditions
        ),
    }


def _replicate_row(replicate: MultiEpochReplicate) -> dict[str, object]:
    return dict(replicate.__dict__)


def _summary_row(summary: MultiEpochSummary) -> dict[str, object]:
    return dict(summary.__dict__)


def _failure_rows(
    summaries: Sequence[MultiEpochSummary],
) -> list[dict[str, object]]:
    failed = [summary for summary in summaries if summary.status == "FAILED"]
    width_failures = [
        summary for summary in summaries if summary.width_68_reduction_fraction <= 0.0
    ]
    return [
        {
            "groups": len(summaries),
            "failed_groups": len(failed),
            "completed_groups": len(summaries) - len(failed),
            "groups_without_68_width_reduction": len(width_failures),
            "failure_causes": "; ".join(sorted({s.failure_cause for s in failed})),
        }
    ]


def _analysis_freeze_text(result: MultiEpochCampaignResult) -> str:
    return "\n".join(
        [
            "# Phase 13.5 Multi-Epoch Analysis Freeze",
            "",
            "Status: frozen before Phase 14 manuscript drafting.",
            "",
            f"Config version: `{result.config.config_version}`.",
            f"Confirmatory config: `{result.config.confirmatory_config_path}`.",
            f"Model backend: `{result.confirmatory_config.model_backend}`.",
            f"Groups: {len(result.groups)}.",
            f"Epochs per group: {result.config.min_epoch_count}.",
            f"Replicates per group: {result.config.replicate_count}.",
            f"Spin grid count: {result.confirmatory_config.spin_grid_count}.",
            f"Energy bin count: {result.confirmatory_config.energy_bin_count}.",
            f"Ray screen size: {result.confirmatory_config.ray_screen_size}x"
            f"{result.confirmatory_config.ray_screen_size}.",
            f"Limb darkening: `{result.confirmatory_config.limb_darkening}`.",
            "",
            "Each group pairs locked Phase 12 conditions with the same spin,",
            "inclination, fitted color correction, true color correction, and",
            "inner-stress setting, while treating the two luminosities as epochs.",
            "The comparison fits each epoch separately and then jointly with a",
            "shared spin parameter and fixed epoch-level luminosity metadata.",
            "",
        ]
    )
