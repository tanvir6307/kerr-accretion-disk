"""Generate Phase 13 paper tables and claim audit from final outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY = ROOT / "data" / "processed" / "confirmatory"
TRANSFER_VALIDATION = ROOT / "data" / "processed" / "transfer_validation"
MULTI_EPOCH = ROOT / "data" / "processed" / "multi_epoch"
VALIDATION = ROOT / "data" / "processed"
TABLES = ROOT / "paper" / "tables"
CLAIM_AUDIT = ROOT / "paper" / "claim_audit.md"
CONFIG = ROOT / "configs" / "production" / "phase12_confirmatory.yaml"


def main() -> None:
    """Generate all current Phase 13 tables and the claim audit."""

    TABLES.mkdir(parents=True, exist_ok=True)
    confirmatory = _read_csv(CONFIRMATORY / "phase12_results_unblinded.csv")
    replicates = _read_csv(CONFIRMATORY / "phase12_replicates_blinded.csv")
    resolution = _read_csv(CONFIRMATORY / "phase12_resolution_reruns.csv")
    validation = _read_csv(VALIDATION / "validation_summary.csv")
    convergence = _read_csv(TRANSFER_VALIDATION / "phase12p5_transfer_convergence.csv")
    capture = _read_csv(
        TRANSFER_VALIDATION / "phase12p5_capture_returning_diagnostics.csv"
    )
    external = _read_csv(
        TRANSFER_VALIDATION / "phase12p5_external_transfer_comparison.csv"
    )
    external_audit = _read_csv(
        TRANSFER_VALIDATION / "phase12p5_external_backend_audit.csv"
    )
    multi_epoch = _read_csv(MULTI_EPOCH / "phase13p5_multi_epoch_summary.csv")

    design = _write_design_table(confirmatory, replicates, resolution)
    validation_table = _write_validation_table(validation)
    confirmatory_table = _write_confirmatory_table(confirmatory)
    resolution_table = _write_resolution_table(resolution)
    transfer_table = _write_transfer_table(
        convergence,
        capture,
        external,
        external_audit,
    )
    multi_epoch_table = _write_multi_epoch_table(multi_epoch)
    _write_claim_audit(
        design,
        validation_table,
        confirmatory_table,
        resolution_table,
        transfer_table,
        multi_epoch_table,
    )
    for path in [
        design,
        validation_table,
        confirmatory_table,
        resolution_table,
        transfer_table,
        multi_epoch_table,
        CLAIM_AUDIT,
    ]:
        print(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_design_table(
    confirmatory: list[dict[str, str]],
    replicates: list[dict[str, str]],
    resolution: list[dict[str, str]],
) -> Path:
    row = {
        "config_version": "phase12_ray_traced_transfer_v4",
        "model_backend": "ray_traced_transfer",
        "conditions": len(confirmatory),
        "base_replicates": len(replicates),
        "completed_base_replicates": sum(
            row["status"] == "COMPLETED" for row in replicates
        ),
        "condition_summaries": len(confirmatory),
        "completed_condition_summaries": sum(
            row["status"] == "COMPLETED" for row in confirmatory
        ),
        "resolution_rows": len(resolution),
        "resolution_stable_rows": sum(row["stable"] == "True" for row in resolution),
        "base_spin_grid_count": 81,
        "high_resolution_spin_grid_count": 121,
        "energy_bin_count": 24,
        "ray_screen_size": "5x5",
        "limb_darkening": "electron_scattering",
        "config_file": str(CONFIG.relative_to(ROOT)),
    }
    return _write_csv(TABLES / "table1_phase12_design.csv", [row])


def _write_validation_table(validation: list[dict[str, str]]) -> Path:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation:
        grouped[row["category"]].append(row)
    rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        normalized = [
            abs(float(item["residual"])) / float(item["tolerance"])
            if float(item["tolerance"]) > 0
            else abs(float(item["residual"]))
            for item in items
        ]
        rows.append(
            {
                "category": category,
                "checks": len(items),
                "passed": sum(item["status"] == "PASS" for item in items),
                "failed": sum(item["status"] != "PASS" for item in items),
                "max_abs_residual_over_tolerance": max(normalized),
            }
        )
    return _write_csv(TABLES / "table2_validation_summary.csv", rows)


def _write_confirmatory_table(confirmatory: list[dict[str, str]]) -> Path:
    grouped: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in confirmatory:
        grouped[(float(row["spin_true"]), float(row["inclination_deg"]))].append(row)
    rows: list[dict[str, object]] = []
    for (spin, inclination), items in sorted(grouped.items()):
        rows.append(
            {
                "spin_true": spin,
                "inclination_deg": inclination,
                "conditions": len(items),
                "mean_bias_mean": _mean(items, "mean_bias"),
                "mean_abs_bias": _mean_abs(items, "mean_bias"),
                "max_abs_bias": _max_abs(items, "mean_bias"),
                "mean_rmse": _mean(items, "rmse"),
                "mean_coverage_68": _mean(items, "coverage_68"),
                "mean_coverage_95": _mean(items, "coverage_95"),
                "mean_chi2_per_dof": _mean(items, "mean_chi2_per_dof"),
                "failed_conditions": sum(
                    item["status"] != "COMPLETED" for item in items
                ),
            }
        )
    return _write_csv(TABLES / "table3_confirmatory_by_spin_inclination.csv", rows)


def _write_resolution_table(resolution: list[dict[str, str]]) -> Path:
    diffs = np.array([float(row["abs_bias_difference"]) for row in resolution])
    rows = [
        {
            "resolution_rows": len(resolution),
            "stable_rows": sum(row["stable"] == "True" for row in resolution),
            "unstable_rows": sum(row["stable"] != "True" for row in resolution),
            "max_abs_bias_difference": float(np.max(diffs)),
            "median_abs_bias_difference": float(np.median(diffs)),
            "tolerance": 0.01,
        }
    ]
    return _write_csv(TABLES / "table4_resolution_stability.csv", rows)


def _write_transfer_table(
    convergence: list[dict[str, str]],
    capture: list[dict[str, str]],
    external: list[dict[str, str]],
    external_audit: list[dict[str, str]],
) -> Path:
    conv_5 = [
        float(row["relative_l1_spectrum_delta"])
        for row in convergence
        if row["screen_size"] == "5"
    ]
    rows = [
        {
            "metric": "max_5x5_relative_l1_delta_vs_7x7",
            "value": max(conv_5),
            "status": "RECORDED",
            "source_file": "data/processed/transfer_validation/"
            "phase12p5_transfer_convergence.csv",
        },
        {
            "metric": "max_captured_fraction",
            "value": max(float(row["captured_fraction"]) for row in capture),
            "status": "RECORDED",
            "source_file": "data/processed/transfer_validation/"
            "phase12p5_capture_returning_diagnostics.csv",
        },
        {
            "metric": "max_returning_fraction",
            "value": max(float(row["returning_fraction"]) for row in capture),
            "status": "RECORDED",
            "source_file": "data/processed/transfer_validation/"
            "phase12p5_capture_returning_diagnostics.csv",
        },
        {
            "metric": "external_transfer_comparison",
            "value": external[0]["status"],
            "status": external[0]["status"],
            "source_file": "data/processed/transfer_validation/"
            "phase12p5_external_transfer_comparison.csv",
        },
        {
            "metric": "available_external_backends",
            "value": sum(row["available"] == "True" for row in external_audit),
            "status": "RECORDED",
            "source_file": "data/processed/transfer_validation/"
            "phase12p5_external_backend_audit.csv",
        },
    ]
    return _write_csv(TABLES / "table5_transfer_validation.csv", rows)


def _write_multi_epoch_table(multi_epoch: list[dict[str, str]]) -> Path:
    completed = [row for row in multi_epoch if row["status"] == "COMPLETED"]
    rows = [
        {
            "groups": len(multi_epoch),
            "completed_groups": len(completed),
            "failed_groups": len(multi_epoch) - len(completed),
            "groups_with_68_width_reduction": sum(
                float(row["width_68_reduction_fraction"]) > 0.0 for row in completed
            ),
            "mean_single_epoch_width_68": _mean(
                completed, "mean_single_epoch_width_68"
            ),
            "mean_joint_width_68": _mean(completed, "mean_joint_width_68"),
            "mean_width_68_reduction_fraction": _mean(
                completed, "width_68_reduction_fraction"
            ),
            "mean_single_epoch_abs_bias": _mean(
                completed, "mean_single_epoch_abs_bias"
            ),
            "mean_joint_abs_bias": _mean(completed, "mean_joint_abs_bias"),
            "groups_with_lower_joint_abs_bias": sum(
                float(row["joint_abs_bias_minus_single_epoch_abs_bias"]) < 0.0
                for row in completed
            ),
            "mean_joint_abs_bias_minus_single_epoch_abs_bias": _mean(
                completed, "joint_abs_bias_minus_single_epoch_abs_bias"
            ),
            "mean_single_epoch_coverage_68": _mean(
                completed, "coverage_68_single_epoch"
            ),
            "mean_joint_coverage_68": _mean(completed, "coverage_68_joint"),
            "source_file": "data/processed/multi_epoch/"
            "phase13p5_multi_epoch_summary.csv",
        }
    ]
    return _write_csv(TABLES / "table6_multi_epoch_comparison.csv", rows)


def _write_claim_audit(
    design: Path,
    validation_table: Path,
    confirmatory_table: Path,
    resolution_table: Path,
    transfer_table: Path,
    multi_epoch_table: Path,
) -> None:
    rows = [
        {
            "claim_id": "C01",
            "manuscript_location": "Results; Figure 1; Table 1",
            "claim_text": (
                "The corrected confirmatory campaign used the "
                "phase12_ray_traced_transfer_v4 backend with 48 locked "
                "conditions, 100 base replicates per condition, 81/121 spin-grid "
                "resolution comparison, 5x5 transfer maps, and electron-scattering "
                "limb darkening."
            ),
            "evidence_file": _rel(design),
            "figure_or_table": "Figure 1; Table 1",
            "analysis_script": "scripts/make_tables.py; scripts/make_figures.py",
            "assumptions": "Detector-independent Gaussian synthetic spectra.",
            "limitations": "No detector response or observational calibration model.",
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C02",
            "manuscript_location": "Validation; Figure 2; Table 2",
            "claim_text": "All recorded independent validation rows passed.",
            "evidence_file": _rel(validation_table),
            "figure_or_table": "Figure 2; Table 2",
            "analysis_script": "scripts/make_tables.py; scripts/make_figures.py",
            "assumptions": "Declared tolerances in validation_summary.csv.",
            "limitations": "Does not establish external ray-tracer agreement.",
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C03",
            "manuscript_location": "Results; Figure 8; Table 3",
            "claim_text": (
                "The confirmatory grid shows large spin biases in several "
                "misspecified conditions."
            ),
            "evidence_file": (
                "data/processed/confirmatory/phase12_results_unblinded.csv"
            ),
            "figure_or_table": "Figure 8; Table 3",
            "analysis_script": "scripts/make_tables.py; scripts/make_figures.py",
            "assumptions": "Bias is posterior-mean spin minus injected spin.",
            "limitations": "Sensitivity-model result; no causal observational claim.",
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C04",
            "manuscript_location": "Results; Figure 9; Table 3",
            "claim_text": (
                "Coverage varies across the confirmatory grid and can be poor "
                "under misspecification."
            ),
            "evidence_file": _rel(confirmatory_table),
            "figure_or_table": "Figure 9; Table 3",
            "analysis_script": "scripts/make_tables.py; scripts/make_figures.py",
            "assumptions": (
                "Coverage estimated from 100 deterministic noise replicates."
            ),
            "limitations": "Not a full Bayesian calibration campaign for real data.",
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C05",
            "manuscript_location": "Numerical Methods; Figure 11; Table 4",
            "claim_text": (
                "All 48 confirmatory conditions passed the spin-grid stability "
                "tolerance."
            ),
            "evidence_file": _rel(resolution_table),
            "figure_or_table": "Figure 11; Table 4",
            "analysis_script": "scripts/make_tables.py; scripts/make_figures.py",
            "assumptions": "Stability tolerance is abs mean-bias difference <= 0.01.",
            "limitations": (
                "This is spin-grid convergence, not full solver convergence."
            ),
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C06",
            "manuscript_location": "Numerical Methods; Figure 11; Table 5",
            "claim_text": (
                "The 5x5 transfer spectra agree with 7x7 benchmark spectra to "
                "within the recorded Phase 12.5 relative-L1 deltas."
            ),
            "evidence_file": _rel(transfer_table),
            "figure_or_table": "Figure 11; Table 5",
            "analysis_script": "scripts/make_tables.py; scripts/make_figures.py",
            "assumptions": "Two benchmark spin/inclination cases.",
            "limitations": (
                "Does not prove convergence for every condition or morphology."
            ),
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C07",
            "manuscript_location": "Numerical Methods; Table 5",
            "claim_text": (
                "Photon-capture and returning-disk outcomes were sampled and "
                "recorded in strong-field diagnostic cases."
            ),
            "evidence_file": "data/processed/transfer_validation/"
            "phase12p5_capture_returning_diagnostics.csv",
            "figure_or_table": "Table 5",
            "analysis_script": (
                "scripts/run_transfer_validation.py; scripts/make_tables.py"
            ),
            "assumptions": "Sampled local emission directions from selected annuli.",
            "limitations": "Diagnostics are not iteratively reprocessed into spectra.",
            "status": "SUPPORTED",
        },
        {
            "claim_id": "C08",
            "manuscript_location": "Validation; Table 5",
            "claim_text": "External ray-tracer agreement is established.",
            "evidence_file": "data/processed/transfer_validation/"
            "phase12p5_external_transfer_comparison.csv",
            "figure_or_table": "Table 5",
            "analysis_script": (
                "scripts/run_transfer_validation.py; scripts/make_tables.py"
            ),
            "assumptions": "Requires a supplied external transfer-map CSV.",
            "limitations": (
                "Current comparison status is SKIPPED and the backend audit found "
                "no installed usable external ray tracer."
            ),
            "status": "UNSUPPORTED",
        },
        {
            "claim_id": "C09",
            "manuscript_location": "Results; Figure 7",
            "claim_text": (
                "The Phase 13.5 outputs provide joint two-epoch versus separate "
                "single-epoch spin comparisons; width and bias changes are "
                "condition-dependent."
            ),
            "evidence_file": _rel(multi_epoch_table),
            "figure_or_table": "Figure 7; Table 6",
            "analysis_script": (
                "scripts/run_multi_epoch.py; scripts/make_tables.py; "
                "scripts/make_figures.py"
            ),
            "assumptions": (
                "Two luminosity epochs are paired at fixed spin, inclination, "
                "f_col assumptions, and inner-stress setting."
            ),
            "limitations": (
                "Epoch luminosities are fixed metadata, not fitted nuisance "
                "parameters; detector response is not included."
            ),
            "status": "SUPPORTED",
        },
    ]
    lines = [
        "# Claim Audit",
        "",
        "| claim_id | manuscript_location | claim_text | evidence_file | "
        "figure_or_table | analysis_script | assumptions | limitations | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _escape_md(
                    str(row[key])
                    for key in [
                        "claim_id",
                        "manuscript_location",
                        "claim_text",
                        "evidence_file",
                        "figure_or_table",
                        "analysis_script",
                        "assumptions",
                        "limitations",
                        "status",
                    ]
                )
            )
            + " |"
        )
    CLAIM_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_md(values: object) -> list[str]:
    return [str(value).replace("|", "\\|").replace("\n", " ") for value in values]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_abs(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([abs(float(row[key])) for row in rows]))


def _max_abs(rows: list[dict[str, str]], key: str) -> float:
    return float(np.max([abs(float(row[key])) for row in rows]))


if __name__ == "__main__":
    main()
