"""Joint marginalized multi-epoch fit over spectral emulators.

Each epoch is fitted with the color-marginalized joint fit used by the
confirmatory campaign. The separate-epoch baseline fits each epoch alone; the
joint fit shares one spin and one color correction across the paired epochs by
concatenating their spectra into a single likelihood. Outputs reuse the
multi-epoch summary schema so the existing figure and table generators apply.
"""

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from kerrdisk.confirmatory import (
    ConfirmatoryConfig,
    TransferCache,
    load_confirmatory_config,
    load_locked_conditions,
)
from kerrdisk.emulator import SpectralGrid
from kerrdisk.inference import ParameterSpec, UniformPrior, credible_interval
from kerrdisk.joint_campaign import (
    JointCampaignConfig,
    _condition_emulator,
    _true_condition_flux,
    validate_joint_campaign_config,
)
from kerrdisk.joint_inference import (
    ForwardModel,
    flatten_chain,
    initial_walker_ball,
    make_spectrum_posterior,
    run_emcee_sampler,
)
from kerrdisk.multi_epoch import (
    MultiEpochCampaignResult,
    MultiEpochConfig,
    MultiEpochGroup,
    MultiEpochReplicate,
    MultiEpochSummary,
    build_multi_epoch_groups,
    summarize_multi_epoch_group,
    write_multi_epoch_outputs,
)
from kerrdisk.synthetic import derive_seed, make_log_energy_bins

type FloatArray = NDArray[np.float64]


def run_joint_multi_epoch_campaign(
    config: JointCampaignConfig,
    *,
    min_epoch_count: int = 2,
    verbose: bool = False,
    transfer_cache: TransferCache | None = None,
    emulator_cache: dict[tuple[float, float], SpectralGrid] | None = None,
) -> MultiEpochCampaignResult:
    """Run the color-marginalized separate-versus-joint multi-epoch comparison.

    ``transfer_cache`` and ``emulator_cache`` may be supplied to reuse the maps
    and emulators already built by a companion confirmatory campaign.
    """

    validate_joint_campaign_config(config)
    conf = load_confirmatory_config(config.confirmatory_config_path)
    if conf.model_backend != "ray_traced_transfer":
        msg = "joint multi-epoch campaign requires the ray_traced_transfer backend"
        raise ValueError(msg)
    conditions = load_locked_conditions(
        conf.locked_conditions_path, master_seed=conf.master_seed
    )
    groups = build_multi_epoch_groups(conditions, min_epoch_count=min_epoch_count)
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

    if transfer_cache is None:
        transfer_cache = {}
    if emulator_cache is None:
        emulator_cache = {}
    replicates: list[MultiEpochReplicate] = []
    summaries: list[MultiEpochSummary] = []
    for index, group in enumerate(groups, start=1):
        if verbose:
            print(  # noqa: T201
                f"[joint-multi] group {index}/{len(groups)} {group.group_id} "
                f"(a*={group.spin_true:g}, i={group.inclination_deg:g}, "
                f"maps cached={len(transfer_cache)})",
                flush=True,
            )
        epoch_emulators = [
            _condition_emulator(
                conf,
                condition,
                spin_nodes,
                f_col_nodes,
                bins,
                transfer_cache,
                emulator_cache,
            )
            for condition in group.conditions
        ]
        epoch_true = [
            _true_condition_flux(conf, condition, bins, transfer_cache)
            for condition in group.conditions
        ]
        group_replicates = tuple(
            _run_joint_multi_epoch_replicate(
                config,
                conf,
                group,
                prior,
                epoch_emulators,
                epoch_true,
                replicate_index,
            )
            for replicate_index in range(config.replicate_count)
        )
        replicates.extend(group_replicates)
        summaries.append(summarize_multi_epoch_group(group, group_replicates))

    multi_config = MultiEpochConfig(
        config_version="phase13p5_joint_emulator_v5",
        confirmatory_config_path=config.confirmatory_config_path,
        master_seed=config.master_seed,
        replicate_count=config.replicate_count,
        min_epoch_count=min_epoch_count,
        width_reduction_required_fraction=0.0,
    )
    return MultiEpochCampaignResult(
        config=multi_config,
        confirmatory_config=conf,
        groups=groups,
        replicates=tuple(replicates),
        summaries=tuple(summaries),
    )


def _run_joint_multi_epoch_replicate(
    config: JointCampaignConfig,
    conf: ConfirmatoryConfig,
    group: MultiEpochGroup,
    prior: UniformPrior,
    epoch_emulators: Sequence[SpectralGrid],
    epoch_true: Sequence[FloatArray],
    replicate_index: int,
) -> MultiEpochReplicate:
    started = perf_counter()
    noise_seed = derive_seed(
        config.master_seed, group.group_id, replicate_index, "multi_epoch_joint"
    )
    try:
        observed: list[FloatArray] = []
        variances: list[FloatArray] = []
        for epoch_index, true_flux in enumerate(epoch_true):
            epoch_seed = derive_seed(noise_seed, "epoch", epoch_index)
            rng = np.random.default_rng(epoch_seed)
            sigma = conf.gaussian_relative_error * true_flux
            observed.append(rng.normal(loc=true_flux, scale=sigma))
            variances.append(sigma * sigma)

        single_fits = [
            _fit_forward(config, prior, emulator.evaluate, obs, var, noise_seed, tag)
            for tag, (emulator, obs, var) in enumerate(
                zip(epoch_emulators, observed, variances, strict=True)
            )
        ]

        def joint_forward(parameters: FloatArray) -> FloatArray:
            return np.concatenate(
                [
                    np.asarray(emulator.evaluate(parameters), dtype=np.float64)
                    for emulator in epoch_emulators
                ]
            )

        joint_fit = _fit_forward(
            config,
            prior,
            joint_forward,
            np.concatenate(observed),
            np.concatenate(variances),
            noise_seed,
            tag=99,
        )

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


def _fit_forward(
    config: JointCampaignConfig,
    prior: UniformPrior,
    forward: ForwardModel,
    observed: FloatArray,
    variance: FloatArray,
    seed: int,
    tag: int,
) -> dict[str, float]:
    problem = make_spectrum_posterior(prior, forward, observed, variance)
    center = 0.5 * (prior.lower_bounds + prior.upper_bounds)
    positions = initial_walker_ball(
        prior,
        center,
        walker_count=config.walker_count,
        scale_fraction=0.15,
        seed=int((seed + tag) % (2**31 - 1)),
    )
    chain = run_emcee_sampler(
        problem,
        initial_positions=positions,
        draws=config.draws,
        burn_in=config.burn_in,
        seed=int((seed // 13 + tag) % (2**31 - 1)),
    )
    spin_samples = flatten_chain(chain)[:, 0]
    ci68 = credible_interval(spin_samples, 0.68)
    ci95 = credible_interval(spin_samples, 0.95)
    return {
        "spin_mean": float(np.mean(spin_samples)),
        "ci68_lower": ci68[0],
        "ci68_upper": ci68[1],
        "ci95_lower": ci95[0],
        "ci95_upper": ci95[1],
    }


def _fractional_reduction(single_value: float, joint_value: float) -> float:
    if single_value <= 0.0:
        return float("nan")
    return float(1.0 - (joint_value / single_value))


def write_joint_multi_epoch_outputs(
    result: MultiEpochCampaignResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write joint multi-epoch outputs using the multi-epoch schema."""

    return write_multi_epoch_outputs(result, output_dir)
