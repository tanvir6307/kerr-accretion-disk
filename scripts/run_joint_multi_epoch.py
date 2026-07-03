"""Run the joint marginalized multi-epoch v5 comparison and write outputs."""

from pathlib import Path

from kerrdisk.joint_campaign import JointCampaignConfig
from kerrdisk.joint_multi_epoch import (
    run_joint_multi_epoch_campaign,
    write_joint_multi_epoch_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "multi_epoch"


def main() -> None:
    """Run the reduced-resolution joint multi-epoch v5 comparison."""

    config = JointCampaignConfig(
        config_version="phase13p5_joint_emulator_v5",
        confirmatory_config_path=(
            ROOT / "configs" / "production" / "phase12_joint_v5.yaml"
        ),
        master_seed=20260623,
        replicate_count=30,
        spin_node_count=9,
        f_col_node_count=5,
        spin_fit_min=-0.9,
        spin_fit_max=0.95,
        f_col_fit_min=1.4,
        f_col_fit_max=2.2,
        walker_count=16,
        draws=350,
        burn_in=200,
    )
    result = run_joint_multi_epoch_campaign(config, verbose=True)
    paths = write_joint_multi_epoch_outputs(result, OUTPUT_DIR)
    completed = sum(1 for s in result.summaries if s.status == "COMPLETED")
    width_reduced = sum(
        1 for s in result.summaries if s.width_68_reduction_fraction > 0.0
    )
    print(f"[joint-multi] groups completed: {completed}/{len(result.summaries)}")
    print(f"[joint-multi] groups with 68% width reduction: {width_reduced}")
    print(f"[joint-multi] wrote summary to {paths['summary']}")


if __name__ == "__main__":
    main()
