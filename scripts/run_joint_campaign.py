"""Run the joint marginalized-fit v5 confirmatory campaign and write outputs."""

from pathlib import Path

from kerrdisk.joint_campaign import (
    JointCampaignConfig,
    run_joint_confirmatory_campaign,
    write_joint_campaign_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "confirmatory"


def main() -> None:
    """Run the reduced-resolution joint v5 confirmatory campaign."""

    config = JointCampaignConfig(
        config_version="phase12_joint_emulator_v5",
        confirmatory_config_path=(
            ROOT / "configs" / "production" / "phase12_joint_v5.yaml"
        ),
        master_seed=20260622,
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
    result = run_joint_confirmatory_campaign(config, verbose=True)
    paths = write_joint_campaign_outputs(result, OUTPUT_DIR)
    completed = sum(1 for s in result.summaries if s.status == "COMPLETED")
    print(f"[joint] conditions completed: {completed}/{len(result.summaries)}")
    print(f"[joint] wrote results to {paths['results_unblinded']}")


if __name__ == "__main__":
    main()
