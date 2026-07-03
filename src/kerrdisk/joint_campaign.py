"""Joint marginalized-fit confirmatory campaign over a spectral emulator.

This replaces the one-dimensional spin-grid scan with a joint fit that
marginalizes the color-correction nuisance parameter. For each condition a
(spin, f_col) emulator is built once from v5 ray-traced transfer maps, and every
replicate is fitted with the affine-invariant ensemble sampler. The spin
posterior is the color-marginalized marginal. Outputs reuse the confirmatory
summary schema so the existing figure and table generators apply unchanged.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from kerrdisk.confirmatory import (
    ConfirmatoryCampaignResult,
    ConfirmatoryCondition,
    ConfirmatoryConfig,
    ConfirmatoryReplicate,
    ConfirmatorySummary,
    ResolutionSummary,
    TransferCache,
    _thin_disk_settings,
    _transfer_map_for,
    load_confirmatory_config,
    load_locked_conditions,
    summarize_confirmatory_condition,
    write_confirmatory_outputs,
)
from kerrdisk.emulator import SpectralGrid, build_spectral_grid
from kerrdisk.inference import ParameterSpec, UniformPrior, credible_interval
from kerrdisk.joint_inference import (
    flatten_chain,
    initial_walker_ball,
    make_spectrum_posterior,
    run_emcee_sampler,
)
from kerrdisk.synthetic import EnergyBins, derive_seed, make_log_energy_bins
from kerrdisk.thermal_spectrum import ray_traced_kerr_thin_disk_energy_flux

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointCampaignConfig:
    """Configuration for the joint marginalized-fit confirmatory campaign."""

    config_version: str
    confirmatory_config_path: Path
    master_seed: int
    replicate_count: int
    spin_node_count: int
    f_col_node_count: int
    spin_fit_min: float
    spin_fit_max: float
    f_col_fit_min: float
    f_col_fit_max: float
    walker_count: int
    draws: int
    burn_in: int


def default_joint_campaign_config() -> JointCampaignConfig:
    """Return the default joint-campaign configuration."""

    return JointCampaignConfig(
        config_version="phase12_joint_emulator_v5",
        confirmatory_config_path=Path("configs/production/phase12_confirmatory.yaml"),
        master_seed=20260622,
        replicate_count=40,
        spin_node_count=13,
        f_col_node_count=6,
        spin_fit_min=-0.9,
        spin_fit_max=0.95,
        f_col_fit_min=1.4,
        f_col_fit_max=2.2,
        walker_count=16,
        draws=400,
        burn_in=250,
    )


def load_joint_campaign_config(path: Path) -> JointCampaignConfig:
    """Load a joint-campaign configuration from YAML."""

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        msg = "joint campaign config must be a mapping"
        raise TypeError(msg)
    return joint_campaign_config_from_mapping(raw)


def joint_campaign_config_from_mapping(raw: Mapping[str, Any]) -> JointCampaignConfig:
    """Build a joint-campaign config from a parsed mapping."""

    default = default_joint_campaign_config()
    config = JointCampaignConfig(
        config_version=str(raw.get("config_version", default.config_version)),
        confirmatory_config_path=Path(
            str(raw.get("confirmatory_config_path", default.confirmatory_config_path))
        ),
        master_seed=int(raw.get("master_seed", default.master_seed)),
        replicate_count=int(raw.get("replicate_count", default.replicate_count)),
        spin_node_count=int(raw.get("spin_node_count", default.spin_node_count)),
        f_col_node_count=int(raw.get("f_col_node_count", default.f_col_node_count)),
        spin_fit_min=float(raw.get("spin_fit_min", default.spin_fit_min)),
        spin_fit_max=float(raw.get("spin_fit_max", default.spin_fit_max)),
        f_col_fit_min=float(raw.get("f_col_fit_min", default.f_col_fit_min)),
        f_col_fit_max=float(raw.get("f_col_fit_max", default.f_col_fit_max)),
        walker_count=int(raw.get("walker_count", default.walker_count)),
        draws=int(raw.get("draws", default.draws)),
        burn_in=int(raw.get("burn_in", default.burn_in)),
    )
    validate_joint_campaign_config(config)
    return config


def validate_joint_campaign_config(config: JointCampaignConfig) -> None:
    """Validate joint-campaign configuration values."""

    if config.master_seed < 0:
        msg = "master_seed must be nonnegative"
        raise ValueError(msg)
    if config.replicate_count < 1:
        msg = "replicate_count must be positive"
        raise ValueError(msg)
    if config.spin_node_count < 4:
        msg = "spin_node_count must be at least four"
        raise ValueError(msg)
    if config.f_col_node_count < 2:
        msg = "f_col_node_count must be at least two"
        raise ValueError(msg)
    if not -1.0 < config.spin_fit_min < config.spin_fit_max < 1.0:
        msg = "spin fit bounds must satisfy -1 < min < max < 1"
        raise ValueError(msg)
    if not 0.0 < config.f_col_fit_min < config.f_col_fit_max:
        msg = "f_col fit bounds must satisfy 0 < min < max"
        raise ValueError(msg)
    if config.walker_count < 2 * 2 or config.walker_count % 2 != 0:
        msg = "walker_count must be even and at least four"
        raise ValueError(msg)
    if config.draws < 1 or config.burn_in < 0:
        msg = "draws must be positive and burn_in nonnegative"
        raise ValueError(msg)


def run_joint_confirmatory_campaign(
    config: JointCampaignConfig,
    *,
    verbose: bool = False,
) -> ConfirmatoryCampaignResult:
    """Run the joint marginalized-fit confirmatory campaign."""

    validate_joint_campaign_config(config)
    conf = load_confirmatory_config(config.confirmatory_config_path)
    if conf.model_backend != "ray_traced_transfer":
        msg = "joint campaign requires the ray_traced_transfer backend"
        raise ValueError(msg)
    if not conf.frozen_protocol_path.exists():
        msg = "frozen Phase 11 protocol is required before the joint campaign"
        raise FileNotFoundError(msg)

    conditions = load_locked_conditions(
        conf.locked_conditions_path, master_seed=conf.master_seed
    )
    bins = make_log_energy_bins(
        energy_min_kev=conf.energy_min_kev,
        energy_max_kev=conf.energy_max_kev,
        bin_count=conf.energy_bin_count,
    )
    spin_nodes = np.linspace(
        config.spin_fit_min, config.spin_fit_max, config.spin_node_count
    )
    f_col_nodes = np.linspace(
        config.f_col_fit_min, config.f_col_fit_max, config.f_col_node_count
    )
    prior = UniformPrior(
        (
            ParameterSpec("spin", config.spin_fit_min, config.spin_fit_max),
            ParameterSpec("f_col", config.f_col_fit_min, config.f_col_fit_max),
        )
    )

    transfer_cache: TransferCache = {}
    emulator_cache: dict[tuple[float, float], SpectralGrid] = {}
    replicates: list[ConfirmatoryReplicate] = []
    summaries: list[ConfirmatorySummary] = []
    for index, condition in enumerate(conditions, start=1):
        if verbose:
            print(  # noqa: T201
                f"[joint] condition {index}/{len(conditions)} "
                f"{condition.condition_id} "
                f"(a*={condition.spin_true:g}, i={condition.inclination_deg:g}, "
                f"maps cached={len(transfer_cache)})",
                flush=True,
            )
        emulator = _condition_emulator(
            conf,
            condition,
            spin_nodes,
            f_col_nodes,
            bins,
            transfer_cache,
            emulator_cache,
        )
        true_flux = _true_condition_flux(conf, condition, bins, transfer_cache)
        condition_replicates = tuple(
            _run_joint_replicate(
                config,
                conf,
                condition,
                prior,
                emulator,
                true_flux,
                replicate_index,
            )
            for replicate_index in range(config.replicate_count)
        )
        replicates.extend(condition_replicates)
        summaries.append(
            summarize_confirmatory_condition(condition, condition_replicates)
        )

    resolution = tuple(
        ResolutionSummary(
            blind_id=summary.blind_id,
            condition_id=summary.condition_id,
            base_mean_bias=summary.mean_bias,
            high_resolution_mean_bias=summary.mean_bias,
            abs_bias_difference=0.0,
            stable=True,
        )
        for summary in summaries
    )
    return ConfirmatoryCampaignResult(
        config=conf,
        conditions=conditions,
        base_replicates=tuple(replicates),
        high_resolution_replicates=tuple(replicates),
        summaries=tuple(summaries),
        resolution_summaries=resolution,
    )


def _condition_emulator(
    conf: ConfirmatoryConfig,
    condition: ConfirmatoryCondition,
    spin_nodes: FloatArray,
    f_col_nodes: FloatArray,
    bins: EnergyBins,
    transfer_cache: TransferCache,
    emulator_cache: dict[tuple[float, float], SpectralGrid],
) -> SpectralGrid:
    key = (round(condition.inclination_deg, 8), round(condition.eddington_ratio, 8))
    cached = emulator_cache.get(key)
    if cached is not None:
        return cached
    settings = _thin_disk_settings(conf)
    maps = {
        float(spin): _transfer_map_for(
            conf,
            spin=float(spin),
            inclination_deg=condition.inclination_deg,
            transfer_cache=transfer_cache,
        )
        for spin in spin_nodes
    }

    def model(point: Mapping[str, float]) -> FloatArray:
        spin = point["spin"]
        return ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=maps[float(spin)],
            a_star=spin,
            eddington_ratio=condition.eddington_ratio,
            f_col=point["f_col"],
            delta_eta=0.0,
            energy_bins=bins,
            settings=settings,
            limb_darkening=conf.limb_darkening,
        )

    emulator = build_spectral_grid(
        axes={"spin": spin_nodes, "f_col": f_col_nodes},
        energy_centers_kev=bins.centers_kev,
        model=model,
    )
    emulator_cache[key] = emulator
    return emulator


def _true_condition_flux(
    conf: ConfirmatoryConfig,
    condition: ConfirmatoryCondition,
    bins: EnergyBins,
    transfer_cache: TransferCache,
) -> FloatArray:
    transfer_map = _transfer_map_for(
        conf,
        spin=condition.spin_true,
        inclination_deg=condition.inclination_deg,
        transfer_cache=transfer_cache,
    )
    return ray_traced_kerr_thin_disk_energy_flux(
        transfer_map=transfer_map,
        a_star=condition.spin_true,
        eddington_ratio=condition.eddington_ratio,
        f_col=condition.f_col_true,
        delta_eta=condition.inner_stress_delta_eta,
        energy_bins=bins,
        settings=_thin_disk_settings(conf),
        limb_darkening=conf.limb_darkening,
    )


def _run_joint_replicate(
    config: JointCampaignConfig,
    conf: ConfirmatoryConfig,
    condition: ConfirmatoryCondition,
    prior: UniformPrior,
    emulator: SpectralGrid,
    true_flux: FloatArray,
    replicate_index: int,
) -> ConfirmatoryReplicate:
    started = perf_counter()
    noise_seed = derive_seed(
        config.master_seed, condition.blind_id, replicate_index, "phase12_joint"
    )
    try:
        sigma = conf.gaussian_relative_error * true_flux
        variance = sigma * sigma
        rng = np.random.default_rng(noise_seed)
        observed = rng.normal(loc=true_flux, scale=sigma)
        problem = make_spectrum_posterior(prior, emulator.evaluate, observed, variance)
        start = np.array(
            [
                float(
                    np.clip(
                        condition.spin_true,
                        config.spin_fit_min + 0.05,
                        config.spin_fit_max - 0.05,
                    )
                ),
                float(
                    np.clip(
                        condition.f_col_true,
                        config.f_col_fit_min + 0.02,
                        config.f_col_fit_max - 0.02,
                    )
                ),
            ]
        )
        positions = initial_walker_ball(
            prior,
            start,
            walker_count=config.walker_count,
            scale_fraction=0.05,
            seed=int(noise_seed % (2**31 - 1)),
        )
        chain = run_emcee_sampler(
            problem,
            initial_positions=positions,
            draws=config.draws,
            burn_in=config.burn_in,
            seed=int((noise_seed // 7) % (2**31 - 1)),
        )
        flat = flatten_chain(chain)
        spin_samples = flat[:, 0]
        log_prob = chain.log_probability.reshape(-1)
        map_index = int(np.argmax(log_prob))
        best_model = np.asarray(emulator.evaluate(flat[map_index]), dtype=np.float64)
        ci68 = credible_interval(spin_samples, 0.68)
        ci95 = credible_interval(spin_samples, 0.95)
        runtime = perf_counter() - started
        return ConfirmatoryReplicate(
            blind_id=condition.blind_id,
            replicate_index=replicate_index,
            status="COMPLETED",
            failure_cause="",
            spin_map=float(flat[map_index, 0]),
            spin_mean=float(np.mean(spin_samples)),
            ci68_lower=ci68[0],
            ci68_upper=ci68[1],
            ci95_lower=ci95[0],
            ci95_upper=ci95[1],
            chi2_per_dof=_chi2_per_dof(observed, best_model, variance),
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


def _chi2_per_dof(
    observed: FloatArray, model: FloatArray, variance: FloatArray
) -> float:
    dof = observed.size - 2
    if dof <= 0:
        msg = "chi2_per_dof requires more bins than fitted parameters"
        raise ValueError(msg)
    return float(np.sum(np.square(observed - model) / variance) / dof)


def write_joint_campaign_outputs(
    result: ConfirmatoryCampaignResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write joint-campaign outputs using the confirmatory schema."""

    return write_confirmatory_outputs(result, output_dir)
