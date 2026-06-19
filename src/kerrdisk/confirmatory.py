"""Phase 12 confirmatory-campaign utilities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from math import radians
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from kerrdisk.screening import (
    ScreeningCondition,
    screening_proxy_photon_flux,
)
from kerrdisk.spectrum import TransferMap, build_transfer_map
from kerrdisk.synthetic import (
    EnergyBins,
    derive_seed,
    make_log_energy_bins,
)
from kerrdisk.thermal_spectrum import (
    KerrThinDiskSettings,
    LimbDarkeningMode,
    kerr_thin_disk_energy_flux,
    ray_traced_kerr_thin_disk_energy_flux,
)

type FloatArray = NDArray[np.float64]
type ReplicateStatus = Literal["COMPLETED", "FAILED"]
type ModelBackend = Literal["kerr_thin_disk", "proxy", "ray_traced_transfer"]
type TransferCache = dict[tuple[float, float, int], TransferMap]


@dataclass(frozen=True)
class ConfirmatoryConfig:
    """Phase 12 locked confirmatory-campaign configuration."""

    config_version: str
    locked_conditions_path: Path
    frozen_protocol_path: Path
    master_seed: int
    replicate_count: int
    energy_min_kev: float
    energy_max_kev: float
    energy_bin_count: int
    spin_grid_count: int
    resolution_spin_grid_count: int
    gaussian_relative_error: float
    bias_stability_abs: float
    model_backend: ModelBackend
    radial_grid_count: int
    disk_outer_radius_rg: float
    temperature_scale_kev: float
    ray_screen_alpha_min: float
    ray_screen_alpha_max: float
    ray_screen_beta_min: float
    ray_screen_beta_max: float
    ray_screen_size: int
    ray_observer_radius: float
    ray_disk_outer_radius: float
    ray_step_size: float
    ray_max_steps: int
    ray_escape_radius: float
    limb_darkening: LimbDarkeningMode


@dataclass(frozen=True)
class ConfirmatoryCondition:
    """One locked condition with a blinded analysis identifier."""

    blind_id: str
    condition_id: str
    spin_true: float
    inclination_deg: float
    eddington_ratio: float
    f_col_true: float
    f_col_fit: float
    inner_stress_delta_eta: float


@dataclass(frozen=True)
class ConfirmatoryReplicate:
    """One Phase 12 replicate fit."""

    blind_id: str
    replicate_index: int
    status: ReplicateStatus
    failure_cause: str
    spin_map: float
    spin_mean: float
    ci68_lower: float
    ci68_upper: float
    ci95_lower: float
    ci95_upper: float
    chi2_per_dof: float
    runtime_s: float
    noise_seed: int


@dataclass(frozen=True)
class ConfirmatorySummary:
    """Unblinded Phase 12 condition summary."""

    blind_id: str
    condition_id: str
    status: ReplicateStatus
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


@dataclass(frozen=True)
class ResolutionSummary:
    """Base versus higher-resolution confirmatory comparison."""

    blind_id: str
    condition_id: str
    base_mean_bias: float
    high_resolution_mean_bias: float
    abs_bias_difference: float
    stable: bool


@dataclass(frozen=True)
class ConfirmatoryCampaignResult:
    """Full Phase 12 campaign result."""

    config: ConfirmatoryConfig
    conditions: tuple[ConfirmatoryCondition, ...]
    base_replicates: tuple[ConfirmatoryReplicate, ...]
    high_resolution_replicates: tuple[ConfirmatoryReplicate, ...]
    summaries: tuple[ConfirmatorySummary, ...]
    resolution_summaries: tuple[ResolutionSummary, ...]


def default_confirmatory_config() -> ConfirmatoryConfig:
    """Return the default Phase 12 confirmatory configuration."""

    return ConfirmatoryConfig(
        config_version="phase12_ray_traced_transfer_v4",
        locked_conditions_path=Path(
            "data/processed/screening/phase11_confirmatory_conditions.csv"
        ),
        frozen_protocol_path=Path(
            "data/processed/screening/phase11_confirmatory_protocol.md"
        ),
        master_seed=20260620,
        replicate_count=100,
        energy_min_kev=0.1,
        energy_max_kev=20.0,
        energy_bin_count=24,
        spin_grid_count=81,
        resolution_spin_grid_count=121,
        gaussian_relative_error=0.03,
        bias_stability_abs=0.01,
        model_backend="ray_traced_transfer",
        radial_grid_count=72,
        disk_outer_radius_rg=80.0,
        temperature_scale_kev=20.0,
        ray_screen_alpha_min=-8.0,
        ray_screen_alpha_max=8.0,
        ray_screen_beta_min=35.0,
        ray_screen_beta_max=65.0,
        ray_screen_size=5,
        ray_observer_radius=100.0,
        ray_disk_outer_radius=80.0,
        ray_step_size=0.1,
        ray_max_steps=3_000,
        ray_escape_radius=160.0,
        limb_darkening="electron_scattering",
    )


def load_confirmatory_config(path: Path) -> ConfirmatoryConfig:
    """Load Phase 12 config from YAML."""

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        msg = "confirmatory config must be a mapping"
        raise TypeError(msg)
    return confirmatory_config_from_mapping(raw)


def confirmatory_config_from_mapping(raw: Mapping[str, Any]) -> ConfirmatoryConfig:
    """Build a Phase 12 config from a parsed mapping."""

    default = default_confirmatory_config()
    config = ConfirmatoryConfig(
        config_version=str(raw.get("config_version", default.config_version)),
        locked_conditions_path=Path(
            str(raw.get("locked_conditions_path", default.locked_conditions_path))
        ),
        frozen_protocol_path=Path(
            str(raw.get("frozen_protocol_path", default.frozen_protocol_path))
        ),
        master_seed=int(raw.get("master_seed", default.master_seed)),
        replicate_count=int(raw.get("replicate_count", default.replicate_count)),
        energy_min_kev=float(raw.get("energy_min_kev", default.energy_min_kev)),
        energy_max_kev=float(raw.get("energy_max_kev", default.energy_max_kev)),
        energy_bin_count=int(raw.get("energy_bin_count", default.energy_bin_count)),
        spin_grid_count=int(raw.get("spin_grid_count", default.spin_grid_count)),
        resolution_spin_grid_count=int(
            raw.get("resolution_spin_grid_count", default.resolution_spin_grid_count)
        ),
        gaussian_relative_error=float(
            raw.get("gaussian_relative_error", default.gaussian_relative_error)
        ),
        bias_stability_abs=float(
            raw.get("bias_stability_abs", default.bias_stability_abs)
        ),
        model_backend=_model_backend_value(
            raw.get("model_backend", default.model_backend)
        ),
        radial_grid_count=int(raw.get("radial_grid_count", default.radial_grid_count)),
        disk_outer_radius_rg=float(
            raw.get("disk_outer_radius_rg", default.disk_outer_radius_rg)
        ),
        temperature_scale_kev=float(
            raw.get("temperature_scale_kev", default.temperature_scale_kev)
        ),
        ray_screen_alpha_min=float(
            raw.get("ray_screen_alpha_min", default.ray_screen_alpha_min)
        ),
        ray_screen_alpha_max=float(
            raw.get("ray_screen_alpha_max", default.ray_screen_alpha_max)
        ),
        ray_screen_beta_min=float(
            raw.get("ray_screen_beta_min", default.ray_screen_beta_min)
        ),
        ray_screen_beta_max=float(
            raw.get("ray_screen_beta_max", default.ray_screen_beta_max)
        ),
        ray_screen_size=int(raw.get("ray_screen_size", default.ray_screen_size)),
        ray_observer_radius=float(
            raw.get("ray_observer_radius", default.ray_observer_radius)
        ),
        ray_disk_outer_radius=float(
            raw.get("ray_disk_outer_radius", default.ray_disk_outer_radius)
        ),
        ray_step_size=float(raw.get("ray_step_size", default.ray_step_size)),
        ray_max_steps=int(raw.get("ray_max_steps", default.ray_max_steps)),
        ray_escape_radius=float(
            raw.get("ray_escape_radius", default.ray_escape_radius)
        ),
        limb_darkening=_limb_darkening_value(
            raw.get("limb_darkening", default.limb_darkening)
        ),
    )
    validate_confirmatory_config(config)
    return config


def validate_confirmatory_config(config: ConfirmatoryConfig) -> None:
    """Validate Phase 12 config values."""

    if config.master_seed < 0:
        msg = "master_seed must be nonnegative"
        raise ValueError(msg)
    if config.replicate_count < 1:
        msg = "replicate_count must be positive"
        raise ValueError(msg)
    if config.energy_min_kev <= 0.0 or config.energy_max_kev <= config.energy_min_kev:
        msg = "energy bounds must be positive and ordered"
        raise ValueError(msg)
    if config.energy_bin_count < 2:
        msg = "energy_bin_count must be at least two"
        raise ValueError(msg)
    if config.spin_grid_count < 11:
        msg = "spin_grid_count must be at least eleven"
        raise ValueError(msg)
    if config.resolution_spin_grid_count <= config.spin_grid_count:
        msg = "resolution_spin_grid_count must exceed spin_grid_count"
        raise ValueError(msg)
    if config.gaussian_relative_error <= 0.0:
        msg = "gaussian_relative_error must be positive"
        raise ValueError(msg)
    if config.bias_stability_abs <= 0.0:
        msg = "bias_stability_abs must be positive"
        raise ValueError(msg)
    if config.radial_grid_count < 8:
        msg = "radial_grid_count must be at least eight"
        raise ValueError(msg)
    if config.disk_outer_radius_rg <= 0.0:
        msg = "disk_outer_radius_rg must be positive"
        raise ValueError(msg)
    if config.temperature_scale_kev <= 0.0:
        msg = "temperature_scale_kev must be positive"
        raise ValueError(msg)
    if config.ray_screen_alpha_max <= config.ray_screen_alpha_min:
        msg = "ray screen alpha bounds must be ordered"
        raise ValueError(msg)
    if config.ray_screen_beta_max <= config.ray_screen_beta_min:
        msg = "ray screen beta bounds must be ordered"
        raise ValueError(msg)
    if config.ray_screen_size < 2:
        msg = "ray_screen_size must be at least two"
        raise ValueError(msg)
    if config.ray_observer_radius <= 0.0:
        msg = "ray_observer_radius must be positive"
        raise ValueError(msg)
    if config.ray_disk_outer_radius <= 0.0:
        msg = "ray_disk_outer_radius must be positive"
        raise ValueError(msg)
    if config.ray_step_size <= 0.0:
        msg = "ray_step_size must be positive"
        raise ValueError(msg)
    if config.ray_max_steps <= 0:
        msg = "ray_max_steps must be positive"
        raise ValueError(msg)
    if config.ray_escape_radius <= 0.0:
        msg = "ray_escape_radius must be positive"
        raise ValueError(msg)


def _model_backend_value(value: object) -> ModelBackend:
    text = str(value)
    if text not in {"kerr_thin_disk", "proxy", "ray_traced_transfer"}:
        msg = (
            "model_backend must be 'kerr_thin_disk', 'proxy', or 'ray_traced_transfer'"
        )
        raise ValueError(msg)
    return cast(ModelBackend, text)


def _limb_darkening_value(value: object) -> LimbDarkeningMode:
    text = str(value)
    if text not in {"isotropic", "electron_scattering"}:
        msg = "limb_darkening must be 'isotropic' or 'electron_scattering'"
        raise ValueError(msg)
    return cast(LimbDarkeningMode, text)


def load_locked_conditions(
    path: Path, *, master_seed: int
) -> tuple[ConfirmatoryCondition, ...]:
    """Load Phase 11 locked conditions and assign blinded IDs."""

    import csv

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        msg = "locked condition list is empty"
        raise ValueError(msg)
    conditions: list[ConfirmatoryCondition] = []
    for row in rows:
        condition_id = row["condition_id"]
        blind_id = _blind_id(master_seed, condition_id)
        conditions.append(
            ConfirmatoryCondition(
                blind_id=blind_id,
                condition_id=condition_id,
                spin_true=float(row["spin_true"]),
                inclination_deg=float(row["inclination_deg"]),
                eddington_ratio=float(row["eddington_ratio"]),
                f_col_true=float(row["f_col_true"]),
                f_col_fit=float(row["f_col_fit"]),
                inner_stress_delta_eta=float(row["inner_stress_delta_eta"]),
            )
        )
    return tuple(conditions)


def _blind_id(master_seed: int, condition_id: str) -> str:
    digest = blake2b(digest_size=6)
    digest.update(str(master_seed).encode("utf-8"))
    digest.update(b"\0")
    digest.update(condition_id.encode("utf-8"))
    return f"blind_{digest.hexdigest()}"


def run_confirmatory_campaign(config: ConfirmatoryConfig) -> ConfirmatoryCampaignResult:
    """Run Phase 12 base and higher-resolution confirmatory fits."""

    validate_confirmatory_config(config)
    if not config.frozen_protocol_path.exists():
        msg = "frozen Phase 11 protocol is required before Phase 12"
        raise FileNotFoundError(msg)
    conditions = load_locked_conditions(
        config.locked_conditions_path,
        master_seed=config.master_seed,
    )
    base = _run_resolution(config, conditions, config.spin_grid_count)
    high = _run_resolution(config, conditions, config.resolution_spin_grid_count)
    summaries = tuple(
        summarize_confirmatory_condition(condition, _replicates_for(base, condition))
        for condition in conditions
    )
    high_summaries = {
        summary.blind_id: summary
        for summary in (
            summarize_confirmatory_condition(
                condition, _replicates_for(high, condition)
            )
            for condition in conditions
        )
    }
    resolution = tuple(
        ResolutionSummary(
            blind_id=summary.blind_id,
            condition_id=summary.condition_id,
            base_mean_bias=summary.mean_bias,
            high_resolution_mean_bias=high_summaries[summary.blind_id].mean_bias,
            abs_bias_difference=abs(
                summary.mean_bias - high_summaries[summary.blind_id].mean_bias
            ),
            stable=abs(summary.mean_bias - high_summaries[summary.blind_id].mean_bias)
            <= config.bias_stability_abs,
        )
        for summary in summaries
    )
    return ConfirmatoryCampaignResult(
        config=config,
        conditions=conditions,
        base_replicates=base,
        high_resolution_replicates=high,
        summaries=summaries,
        resolution_summaries=resolution,
    )


def _run_resolution(
    config: ConfirmatoryConfig,
    conditions: Sequence[ConfirmatoryCondition],
    spin_grid_count: int,
) -> tuple[ConfirmatoryReplicate, ...]:
    bins = make_log_energy_bins(
        energy_min_kev=config.energy_min_kev,
        energy_max_kev=config.energy_max_kev,
        bin_count=config.energy_bin_count,
    )
    spin_grid = np.linspace(-0.998, 0.998, spin_grid_count)
    results: list[ConfirmatoryReplicate] = []
    transfer_cache: TransferCache = {}
    for condition in conditions:
        model_matrix = _model_energy_flux_matrix(
            config,
            bins,
            condition,
            spin_grid,
            transfer_cache,
        )
        for replicate_index in range(config.replicate_count):
            results.append(
                _run_replicate(
                    config,
                    condition,
                    bins,
                    spin_grid,
                    model_matrix,
                    replicate_index,
                    transfer_cache,
                )
            )
    return tuple(results)


def _run_replicate(
    config: ConfirmatoryConfig,
    condition: ConfirmatoryCondition,
    bins: EnergyBins,
    spin_grid: FloatArray,
    model_matrix: FloatArray,
    replicate_index: int,
    transfer_cache: TransferCache,
) -> ConfirmatoryReplicate:
    started = perf_counter()
    noise_seed = derive_seed(
        config.master_seed,
        condition.blind_id,
        replicate_index,
        "phase12",
    )
    try:
        true_flux = _true_energy_flux(config, bins, condition, transfer_cache)
        sigma = config.gaussian_relative_error * true_flux
        rng = np.random.default_rng(noise_seed)
        observed = rng.normal(loc=true_flux, scale=sigma)
        variance = sigma * sigma
        fit = _fit_spin_grid_vectorized(observed, variance, model_matrix, spin_grid)
        runtime = perf_counter() - started
        return ConfirmatoryReplicate(
            blind_id=condition.blind_id,
            replicate_index=replicate_index,
            status="COMPLETED",
            failure_cause="",
            spin_map=fit["spin_map"],
            spin_mean=fit["spin_mean"],
            ci68_lower=fit["ci68_lower"],
            ci68_upper=fit["ci68_upper"],
            ci95_lower=fit["ci95_lower"],
            ci95_upper=fit["ci95_upper"],
            chi2_per_dof=_chi2_per_dof(observed, fit["best_model"], variance),
            runtime_s=runtime,
            noise_seed=noise_seed,
        )
    except (FloatingPointError, ValueError) as exc:
        runtime = perf_counter() - started
        return ConfirmatoryReplicate(
            blind_id=condition.blind_id,
            replicate_index=replicate_index,
            status="FAILED",
            failure_cause=str(exc),
            spin_map=float("nan"),
            spin_mean=float("nan"),
            ci68_lower=float("nan"),
            ci68_upper=float("nan"),
            ci95_lower=float("nan"),
            ci95_upper=float("nan"),
            chi2_per_dof=float("nan"),
            runtime_s=runtime,
            noise_seed=noise_seed,
        )


def _screening_condition(
    condition: ConfirmatoryCondition, *, fit_model: bool
) -> ScreeningCondition:
    return ScreeningCondition(
        condition_id=condition.condition_id,
        spin_true=condition.spin_true,
        inclination_deg=condition.inclination_deg,
        eddington_ratio=condition.eddington_ratio,
        f_col_true=condition.f_col_fit if fit_model else condition.f_col_true,
        f_col_fit=condition.f_col_fit,
        inner_stress_delta_eta=0.0 if fit_model else condition.inner_stress_delta_eta,
        replicate_count=1,
    )


def _true_energy_flux(
    config: ConfirmatoryConfig,
    bins: EnergyBins,
    condition: ConfirmatoryCondition,
    transfer_cache: TransferCache,
) -> FloatArray:
    return _energy_flux_for_spin(
        config,
        bins,
        condition,
        spin=condition.spin_true,
        fit_model=False,
        transfer_cache=transfer_cache,
    )


def _model_energy_flux_matrix(
    config: ConfirmatoryConfig,
    bins: EnergyBins,
    condition: ConfirmatoryCondition,
    spin_grid: FloatArray,
    transfer_cache: TransferCache,
) -> FloatArray:
    rows = [
        _energy_flux_for_spin(
            config,
            bins,
            condition,
            spin=float(spin),
            fit_model=True,
            transfer_cache=transfer_cache,
        )
        for spin in spin_grid
    ]
    return np.vstack(rows)


def _energy_flux_for_spin(
    config: ConfirmatoryConfig,
    bins: EnergyBins,
    condition: ConfirmatoryCondition,
    *,
    spin: float,
    fit_model: bool,
    transfer_cache: TransferCache,
) -> FloatArray:
    if config.model_backend == "proxy":
        photon_flux = screening_proxy_photon_flux(
            bins,
            _screening_condition(condition, fit_model=fit_model),
            spin,
        )
        return bins.centers_kev * photon_flux
    model_condition = _screening_condition(condition, fit_model=fit_model)
    if config.model_backend == "ray_traced_transfer":
        transfer_map = _transfer_map_for(
            config,
            spin=spin,
            inclination_deg=model_condition.inclination_deg,
            transfer_cache=transfer_cache,
        )
        return ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=transfer_map,
            a_star=spin,
            eddington_ratio=model_condition.eddington_ratio,
            f_col=model_condition.f_col_true,
            delta_eta=model_condition.inner_stress_delta_eta,
            energy_bins=bins,
            settings=KerrThinDiskSettings(
                radial_grid_count=config.radial_grid_count,
                disk_outer_radius_rg=config.disk_outer_radius_rg,
                temperature_scale_kev=config.temperature_scale_kev,
            ),
            limb_darkening=config.limb_darkening,
        )
    return kerr_thin_disk_energy_flux(
        a_star=spin,
        inclination_deg=model_condition.inclination_deg,
        eddington_ratio=model_condition.eddington_ratio,
        f_col=model_condition.f_col_true,
        delta_eta=model_condition.inner_stress_delta_eta,
        energy_bins=bins,
        settings=KerrThinDiskSettings(
            radial_grid_count=config.radial_grid_count,
            disk_outer_radius_rg=config.disk_outer_radius_rg,
            temperature_scale_kev=config.temperature_scale_kev,
        ),
    )


def _transfer_map_for(
    config: ConfirmatoryConfig,
    *,
    spin: float,
    inclination_deg: float,
    transfer_cache: TransferCache,
) -> TransferMap:
    key = (round(spin, 12), round(inclination_deg, 8), config.ray_screen_size)
    cached = transfer_cache.get(key)
    if cached is not None:
        return cached
    alpha = _screen_centers(
        config.ray_screen_alpha_min,
        config.ray_screen_alpha_max,
        config.ray_screen_size,
    )
    beta = _screen_centers(
        config.ray_screen_beta_min,
        config.ray_screen_beta_max,
        config.ray_screen_size,
    )
    transfer_map = build_transfer_map(
        spin,
        alpha,
        beta,
        observer_radius=config.ray_observer_radius,
        observer_theta=radians(inclination_deg),
        disk_outer_radius=config.ray_disk_outer_radius,
        observer_distance=config.ray_observer_radius,
        step_size=config.ray_step_size,
        max_steps=config.ray_max_steps,
        escape_radius=config.ray_escape_radius,
    )
    if transfer_map.emission_radius.size == 0:
        msg = (
            "ray-traced transfer map contains no disk hits for "
            f"spin={spin:.6g}, inclination_deg={inclination_deg:.6g}"
        )
        raise ValueError(msg)
    transfer_cache[key] = transfer_map
    return transfer_map


def _screen_centers(lower: float, upper: float, count: int) -> FloatArray:
    width = (upper - lower) / count
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * width


def _fit_spin_grid_vectorized(
    observed: FloatArray,
    variance: FloatArray,
    model_matrix: FloatArray,
    spin_grid: FloatArray,
) -> dict[str, Any]:
    residual = observed[None, :] - model_matrix
    log_likelihood = -0.5 * np.sum(
        (residual * residual / variance[None, :])
        + np.log(2.0 * np.pi * variance)[None, :],
        axis=1,
    )
    if not np.any(np.isfinite(log_likelihood)):
        msg = "no finite spin-grid likelihood values"
        raise FloatingPointError(msg)
    weights = _normalized_weights(log_likelihood)
    spin_mean = float(np.sum(weights * spin_grid))
    map_index = int(np.argmax(log_likelihood))
    ci68 = _weighted_interval(spin_grid, weights, 0.68)
    ci95 = _weighted_interval(spin_grid, weights, 0.95)
    return {
        "spin_map": float(spin_grid[map_index]),
        "spin_mean": spin_mean,
        "ci68_lower": ci68[0],
        "ci68_upper": ci68[1],
        "ci95_lower": ci95[0],
        "ci95_upper": ci95[1],
        "best_model": model_matrix[map_index],
    }


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


def _chi2_per_dof(
    observed: FloatArray, model: FloatArray, variance: FloatArray
) -> float:
    dof = observed.size - 1
    if dof <= 0:
        msg = "chi2_per_dof requires more bins than fitted parameters"
        raise ValueError(msg)
    return float(np.sum(np.square(observed - model) / variance) / dof)


def _replicates_for(
    replicates: Sequence[ConfirmatoryReplicate],
    condition: ConfirmatoryCondition,
) -> tuple[ConfirmatoryReplicate, ...]:
    return tuple(
        replicate
        for replicate in replicates
        if replicate.blind_id == condition.blind_id
    )


def summarize_confirmatory_condition(
    condition: ConfirmatoryCondition,
    replicates: Sequence[ConfirmatoryReplicate],
) -> ConfirmatorySummary:
    """Return an unblinded condition summary from replicate fits."""

    completed = [
        replicate for replicate in replicates if replicate.status == "COMPLETED"
    ]
    failed = [replicate for replicate in replicates if replicate.status == "FAILED"]
    if completed:
        spin_mean = np.array([replicate.spin_mean for replicate in completed])
        biases = spin_mean - condition.spin_true
        widths68 = np.array(
            [replicate.ci68_upper - replicate.ci68_lower for replicate in completed]
        )
        widths95 = np.array(
            [replicate.ci95_upper - replicate.ci95_lower for replicate in completed]
        )
        chi2 = np.array([replicate.chi2_per_dof for replicate in completed])
        runtimes = np.array([replicate.runtime_s for replicate in completed])
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
        rmse = float(np.sqrt(np.mean(np.square(biases))))
        width68 = float(np.mean(widths68))
        width95 = float(np.mean(widths95))
        mean_chi2 = float(np.mean(chi2))
        runtime = float(np.mean(runtimes))
        cov68 = float(coverage68)
        cov95 = float(coverage95)
    else:
        mean_bias = median_bias = rmse = float("nan")
        width68 = width95 = mean_chi2 = runtime = float("nan")
        cov68 = cov95 = float("nan")
    status: ReplicateStatus = "COMPLETED" if not failed else "FAILED"
    return ConfirmatorySummary(
        blind_id=condition.blind_id,
        condition_id=condition.condition_id,
        status=status,
        failure_cause="; ".join(
            sorted({replicate.failure_cause for replicate in failed})
        ),
        planned_replicates=len(replicates),
        completed_replicates=len(completed),
        failed_replicates=len(failed),
        failure_rate=len(failed) / len(replicates) if replicates else 1.0,
        spin_true=condition.spin_true,
        inclination_deg=condition.inclination_deg,
        eddington_ratio=condition.eddington_ratio,
        f_col_true=condition.f_col_true,
        f_col_fit=condition.f_col_fit,
        inner_stress_delta_eta=condition.inner_stress_delta_eta,
        mean_bias=mean_bias,
        median_bias=median_bias,
        rmse=rmse,
        mean_width_68=width68,
        mean_width_95=width95,
        coverage_68=cov68,
        coverage_95=cov95,
        mean_chi2_per_dof=mean_chi2,
        mean_runtime_s=runtime,
    )


def write_confirmatory_outputs(
    result: ConfirmatoryCampaignResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write Phase 12 blinded, hidden-truth, and unblinded outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "blinded_conditions": output_dir / "phase12_blinded_conditions.csv",
        "hidden_truths": output_dir / "phase12_hidden_truths.csv",
        "analysis_freeze": output_dir / "phase12_analysis_freeze.md",
        "replicates_blinded": output_dir / "phase12_replicates_blinded.csv",
        "results_unblinded": output_dir / "phase12_results_unblinded.csv",
        "resolution_reruns": output_dir / "phase12_resolution_reruns.csv",
        "failure_summary": output_dir / "phase12_failure_summary.csv",
    }
    _write_csv(
        paths["blinded_conditions"],
        [_blinded_condition_row(c) for c in result.conditions],
    )
    _write_csv(
        paths["hidden_truths"], [_hidden_truth_row(c) for c in result.conditions]
    )
    paths["analysis_freeze"].write_text(_analysis_freeze_text(result), encoding="utf-8")
    _write_csv(
        paths["replicates_blinded"], [_replicate_row(r) for r in result.base_replicates]
    )
    _write_csv(paths["results_unblinded"], [_summary_row(s) for s in result.summaries])
    _write_csv(
        paths["resolution_reruns"],
        [_resolution_row(r) for r in result.resolution_summaries],
    )
    _write_csv(paths["failure_summary"], _failure_rows(result.summaries))
    return paths


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _blinded_condition_row(condition: ConfirmatoryCondition) -> dict[str, object]:
    return {
        "blind_id": condition.blind_id,
        "inclination_deg": condition.inclination_deg,
        "eddington_ratio": condition.eddington_ratio,
        "f_col_fit": condition.f_col_fit,
    }


def _hidden_truth_row(condition: ConfirmatoryCondition) -> dict[str, object]:
    return {
        "blind_id": condition.blind_id,
        "condition_id": condition.condition_id,
        "spin_true": condition.spin_true,
        "f_col_true": condition.f_col_true,
        "inner_stress_delta_eta": condition.inner_stress_delta_eta,
    }


def _replicate_row(replicate: ConfirmatoryReplicate) -> dict[str, object]:
    return {
        "blind_id": replicate.blind_id,
        "replicate_index": replicate.replicate_index,
        "status": replicate.status,
        "failure_cause": replicate.failure_cause,
        "spin_map": replicate.spin_map,
        "spin_mean": replicate.spin_mean,
        "ci68_lower": replicate.ci68_lower,
        "ci68_upper": replicate.ci68_upper,
        "ci95_lower": replicate.ci95_lower,
        "ci95_upper": replicate.ci95_upper,
        "chi2_per_dof": replicate.chi2_per_dof,
        "runtime_s": replicate.runtime_s,
        "noise_seed": replicate.noise_seed,
    }


def _summary_row(summary: ConfirmatorySummary) -> dict[str, object]:
    return dict(summary.__dict__)


def _resolution_row(summary: ResolutionSummary) -> dict[str, object]:
    return dict(summary.__dict__)


def _failure_rows(summaries: Sequence[ConfirmatorySummary]) -> list[dict[str, object]]:
    total = len(summaries)
    failed = [summary for summary in summaries if summary.status == "FAILED"]
    return [
        {
            "conditions": total,
            "failed_conditions": len(failed),
            "completed_conditions": total - len(failed),
            "failure_causes": "; ".join(sorted({s.failure_cause for s in failed})),
        }
    ]


def _analysis_freeze_text(result: ConfirmatoryCampaignResult) -> str:
    return "\n".join(
        [
            "# Phase 12 Analysis Freeze",
            "",
            "Status: frozen before hidden truths are joined to final summaries.",
            "",
            f"Config version: `{result.config.config_version}`.",
            f"Locked conditions: `{result.config.locked_conditions_path}`.",
            f"Frozen protocol: `{result.config.frozen_protocol_path}`.",
            f"Replicates per condition: {result.config.replicate_count}.",
            f"Model backend: `{result.config.model_backend}`.",
            f"Base spin grid count: {result.config.spin_grid_count}.",
            "Higher-resolution spin grid count: "
            f"{result.config.resolution_spin_grid_count}.",
            f"Energy bin count: {result.config.energy_bin_count}.",
            f"Bias stability tolerance: {result.config.bias_stability_abs}.",
            f"Ray screen size: {result.config.ray_screen_size}x"
            f"{result.config.ray_screen_size}.",
            "Ray screen alpha range: "
            f"[{result.config.ray_screen_alpha_min}, "
            f"{result.config.ray_screen_alpha_max}].",
            "Ray screen beta range: "
            f"[{result.config.ray_screen_beta_min}, "
            f"{result.config.ray_screen_beta_max}].",
            f"Ray observer radius: {result.config.ray_observer_radius}.",
            f"Ray disk outer radius: {result.config.ray_disk_outer_radius}.",
            f"Ray step size: {result.config.ray_step_size}.",
            f"Ray max steps: {result.config.ray_max_steps}.",
            f"Limb darkening: `{result.config.limb_darkening}`.",
            "",
            "Blinded replicate fits were written before unblinded condition",
            "summaries. The hidden-truth CSV is separate and should not be read",
            "by analysis scripts before this freeze point.",
            "",
        ]
    )
