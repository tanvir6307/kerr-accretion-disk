"""Run the full-resolution v5 joint confirmatory and multi-epoch campaigns.

The two campaigns share one transfer-map and emulator cache, so the expensive
ray-traced maps are built only once.
"""

from pathlib import Path

from kerrdisk.confirmatory import TransferCache
from kerrdisk.emulator import SpectralGrid
from kerrdisk.joint_campaign import (
    JointCampaignConfig,
    run_joint_confirmatory_campaign,
    write_joint_campaign_outputs,
)
from kerrdisk.joint_multi_epoch import (
    run_joint_multi_epoch_campaign,
    write_joint_multi_epoch_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_DIR = ROOT / "data" / "processed" / "confirmatory"
MULTI_EPOCH_DIR = ROOT / "data" / "processed" / "multi_epoch"


def main() -> None:
    """Run both full-resolution v5 campaigns with a shared map cache."""

    config = JointCampaignConfig(
        config_version="phase12_joint_emulator_v5_full",
        confirmatory_config_path=(
            ROOT / "configs" / "production" / "phase12_joint_v5_full.yaml"
        ),
        master_seed=20260622,
        replicate_count=100,
        spin_node_count=13,
        f_col_node_count=9,
        spin_fit_min=-0.9,
        spin_fit_max=0.95,
        f_col_fit_min=1.4,
        f_col_fit_max=2.2,
        walker_count=16,
        draws=400,
        burn_in=250,
    )

    transfer_cache: TransferCache = {}
    emulator_cache: dict[tuple[float, float], SpectralGrid] = {}

    print("[full] confirmatory campaign", flush=True)
    confirmatory = run_joint_confirmatory_campaign(
        config,
        verbose=True,
        transfer_cache=transfer_cache,
        emulator_cache=emulator_cache,
    )
    write_joint_campaign_outputs(confirmatory, CONFIRMATORY_DIR)
    completed = sum(1 for s in confirmatory.summaries if s.status == "COMPLETED")
    print(f"[full] confirmatory conditions completed: {completed}/48", flush=True)

    print("[full] multi-epoch campaign (reusing cached maps)", flush=True)
    multi = run_joint_multi_epoch_campaign(
        config,
        verbose=True,
        transfer_cache=transfer_cache,
        emulator_cache=emulator_cache,
    )
    write_joint_multi_epoch_outputs(multi, MULTI_EPOCH_DIR)
    width_reduced = sum(
        1 for s in multi.summaries if s.width_68_reduction_fraction > 0.0
    )
    print(
        f"[full] multi-epoch groups with 68% width reduction: "
        f"{width_reduced}/{len(multi.summaries)}",
        flush=True,
    )
    print("[full] done", flush=True)


if __name__ == "__main__":
    main()
