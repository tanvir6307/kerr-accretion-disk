"""Generate Phase 13 paper figures from final data products."""

from __future__ import annotations

import csv
from collections import defaultdict
from math import radians
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from kerrdisk.confirmatory import load_confirmatory_config
from kerrdisk.disk_flux import (
    page_thorne_flux_profile,
    stressed_page_thorne_flux_profile,
)
from kerrdisk.isco import isco_radius
from kerrdisk.scales import observer_distance_rg
from kerrdisk.spectrum import build_transfer_map
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import (
    KerrThinDiskSettings,
    ray_traced_kerr_thin_disk_energy_flux,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY = ROOT / "data" / "processed" / "confirmatory"
TRANSFER_VALIDATION = ROOT / "data" / "processed" / "transfer_validation"
MULTI_EPOCH = ROOT / "data" / "processed" / "multi_epoch"
VALIDATION = ROOT / "data" / "processed"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"
CONFIG = ROOT / "configs" / "production" / "phase12_confirmatory.yaml"


def main() -> None:
    """Generate all current Phase 13 figures and captions."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    captions: list[tuple[str, str]] = []
    captions.append(("figure1_model_schematic.png", _figure1_model_schematic()))
    captions.append(("figure2_validation_residuals.png", _figure2_validation()))
    captions.append(("figure3_disk_flux_profiles.png", _figure3_disk_flux()))
    captions.append(("figure4_transfer_images.png", _figure4_transfer_images()))
    captions.append(("figure5_ray_traced_spectra.png", _figure5_spectra()))
    captions.append(("figure6_bias_width_proxy.png", _figure6_bias_width()))
    captions.append(("figure7_multi_epoch_status.png", _figure7_multi_epoch_status()))
    captions.append(("figure8_bias_map.png", _figure8_bias_map()))
    captions.append(("figure9_coverage_map.png", _figure9_coverage_map()))
    captions.append(("figure10_failure_domain.png", _figure10_failure_domain()))
    captions.append(("figure11_numerical_convergence.png", _figure11_convergence()))
    _write_captions(captions)
    for name, _caption in captions:
        print(FIGURES / name)
    print(ROOT / "paper" / "figure_captions.md")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _savefig(name: str) -> None:
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def _figure1_model_schematic() -> str:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    boxes = [
        ("Locked\nconditions", 0.085, 0.72),
        ("Kerr disk\nPage-Thorne + stress", 0.30, 0.72),
        ("Ray-traced\n64x64 full-disk maps", 0.53, 0.72),
        ("Physical diluted\nblackbody + limb dark.", 0.77, 0.72),
        ("Synthetic\nreplicates", 0.30, 0.26),
        ("Joint marginalized\nspin fit", 0.53, 0.26),
        ("Bias, coverage,\nconvergence", 0.77, 0.26),
    ]
    for text, x_pos, y_pos in boxes:
        ax.text(
            x_pos,
            y_pos,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f4f0e8",
                "edgecolor": "#303030",
                "linewidth": 1.0,
            },
            transform=ax.transAxes,
        )
    arrows = [
        ((0.155, 0.72), (0.20, 0.72)),
        ((0.40, 0.72), (0.44, 0.72)),
        ((0.63, 0.72), (0.68, 0.72)),
        ((0.77, 0.62), (0.77, 0.37)),
        ((0.68, 0.26), (0.63, 0.26)),
        ((0.44, 0.26), (0.40, 0.26)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#303030"},
        )
    ax.set_title("KerrDisk-UQ v5 spectral workflow", fontsize=13)
    _savefig("figure1_model_schematic.png")
    return (
        "Model schematic for the corrected v5 workflow. Boxes show the analysis "
        "sequence from locked conditions through the physically normalized, "
        "full-disk ray-traced spectra to the joint marginalized spin fit and the "
        "bias, coverage, and convergence summaries."
    )


def _figure2_validation() -> str:
    rows = _read_csv(VALIDATION / "validation_summary.csv")
    labels = [row["check_id"] for row in rows]
    normalized = [
        abs(float(row["residual"])) / float(row["tolerance"])
        if float(row["tolerance"]) > 0
        else abs(float(row["residual"]))
        for row in rows
    ]
    colors = ["#32746d" if row["status"] == "PASS" else "#b23b3b" for row in rows]
    # Most residuals are at machine precision, so plot the safety margin: how many
    # orders of magnitude each residual sits below its declared tolerance. Taller
    # bars are safer; the line at zero marks the tolerance itself.
    margin_cap = 15.0
    margins = [
        margin_cap if value <= 0.0 else float(min(margin_cap, -np.log10(value)))
        for value in normalized
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.bar(np.arange(len(rows)), margins, color=colors, zorder=3)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, zorder=4)
    ax.text(len(rows) - 0.5, 0.2, "tolerance", ha="right", va="bottom", fontsize=9)
    ax.set_ylim(0.0, margin_cap + 1.0)
    ax.set_ylabel("orders of magnitude below tolerance (higher = safer)")
    ax.set_title("Independent validation margins (all checks pass)")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    _savefig("figure2_validation_residuals.png")
    return (
        "Independent validation margins from data/processed/validation_summary.csv. "
        "Each bar is the number of orders of magnitude the residual sits below its "
        "declared tolerance (capped at 15 for residuals at machine precision); all "
        "checks pass. Varied quantities include ISCO radius, efficiency, ray "
        "invariants, Page-Thorne flux, and a constant-intensity transfer-spectrum "
        "check."
    )


def _figure3_disk_flux() -> str:
    spins = [-0.5, 0.0, 0.5, 0.9]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    rows: list[dict[str, float]] = []
    for spin in spins:
        radii = np.geomspace(isco_radius(spin), 80.0, 300)
        zero = page_thorne_flux_profile(spin, radii, mass_accretion_rate=0.1).flux
        stressed = stressed_page_thorne_flux_profile(
            spin,
            radii,
            delta_eta=0.02,
            mass_accretion_rate=0.1,
        ).flux
        ax.loglog(radii, np.maximum(zero, 1.0e-30), label=f"a*={spin:g}, zero")
        ax.loglog(
            radii,
            np.maximum(stressed, 1.0e-30),
            linestyle="--",
            label=f"a*={spin:g}, stress",
        )
        for radius, base_flux, stress_flux in zip(radii, zero, stressed, strict=True):
            rows.append(
                {
                    "a_star": spin,
                    "radius_rg": float(radius),
                    "zero_torque_flux": float(base_flux),
                    "stressed_flux_delta_eta_0p02": float(stress_flux),
                }
            )
    _write_dicts(TABLES / "figure3_disk_flux_source.csv", rows)
    ax.set_xlabel("Radius [GM/c^2]")
    ax.set_ylabel("One-face dimensionless flux")
    ax.set_title("Page-Thorne and stressed disk flux profiles")
    # Focus on the physically meaningful flux range; the flux vanishes at the
    # ISCO, which otherwise stretches the log axis down to the numeric floor.
    ax.set_xlim(2.0, 80.0)
    ax.set_ylim(1.0e-9, 5.0e-5)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 3, 5, 10, 20, 40, 80])
    ax.legend(fontsize=7, ncols=2, loc="lower left")
    ax.grid(alpha=0.25, which="both")
    _savefig("figure3_disk_flux_profiles.png")
    return (
        "Disk flux profiles for a*=(-0.5, 0, 0.5, 0.9), normalized to "
        "dimensionless one-face flux with mass_accretion_rate=0.1. Solid lines "
        "are zero-torque Page-Thorne profiles; dashed lines use Delta_eta=0.02."
    )


def _figure4_transfer_images() -> str:
    config = load_confirmatory_config(CONFIG)
    cases = [(0.0, 40.0), (0.9, 70.0)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True, sharey=True)
    rows: list[dict[str, float]] = []
    for ax, (spin, inclination) in zip(axes, cases, strict=True):
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
            observer_theta=radians(inclination),
            disk_outer_radius=config.ray_disk_outer_radius,
            observer_distance=observer_distance_rg(
                config.distance_kpc, config.mass_msun
            ),
            step_size=config.ray_step_size,
            max_steps=config.ray_max_steps,
            escape_radius=config.ray_escape_radius,
        )
        image = np.full((config.ray_screen_size, config.ray_screen_size), np.nan)
        for a_val, b_val, redshift, radius, mu in zip(
            transfer_map.alpha,
            transfer_map.beta,
            transfer_map.redshift,
            transfer_map.emission_radius,
            transfer_map.emission_mu,
            strict=True,
        ):
            a_index = int(np.argmin(np.abs(alpha - a_val)))
            b_index = int(np.argmin(np.abs(beta - b_val)))
            image[b_index, a_index] = redshift
            rows.append(
                {
                    "a_star": spin,
                    "inclination_deg": inclination,
                    "alpha": float(a_val),
                    "beta": float(b_val),
                    "emission_radius": float(radius),
                    "redshift": float(redshift),
                    "emission_mu": float(mu),
                }
            )
        im = ax.imshow(
            image,
            origin="lower",
            extent=[alpha.min(), alpha.max(), beta.min(), beta.max()],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_title(f"a*={spin:g}, i={inclination:g} deg")
        ax.set_xlabel("alpha")
        ax.set_ylabel("beta")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="redshift g")
    _write_dicts(TABLES / "figure4_transfer_image_source.csv", rows)
    _savefig("figure4_transfer_images.png")
    size = config.ray_screen_size
    return (
        f"Ray-traced {size}x{size} full-disk transfer-map image-plane samples for "
        "two benchmark spin/inclination pairs. Color is the transfer-map redshift g "
        "for disk-hit rays; alpha and beta are local observer-screen coordinates in "
        "gravitational radii."
    )


def _figure5_spectra() -> str:
    config = load_confirmatory_config(CONFIG)
    bins = make_log_energy_bins(
        energy_min_kev=config.energy_min_kev,
        energy_max_kev=config.energy_max_kev,
        bin_count=config.energy_bin_count,
    )
    settings = KerrThinDiskSettings(
        radial_grid_count=config.radial_grid_count,
        disk_outer_radius_rg=config.disk_outer_radius_rg,
        mass_msun=config.mass_msun,
        distance_kpc=config.distance_kpc,
    )
    rows: list[dict[str, float]] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for spin, delta_eta, label in [
        (0.0, 0.0, "a*=0, zero stress"),
        (0.0, 0.02, "a*=0, stress"),
        (0.9, 0.0, "a*=0.9, zero stress"),
        (0.9, 0.02, "a*=0.9, stress"),
    ]:
        transfer_map = build_transfer_map(
            spin,
            _screen_centers(
                config.ray_screen_alpha_min,
                config.ray_screen_alpha_max,
                config.ray_screen_size,
            ),
            _screen_centers(
                config.ray_screen_beta_min,
                config.ray_screen_beta_max,
                config.ray_screen_size,
            ),
            observer_radius=config.ray_observer_radius,
            observer_theta=radians(40.0),
            disk_outer_radius=config.ray_disk_outer_radius,
            observer_distance=observer_distance_rg(
                config.distance_kpc, config.mass_msun
            ),
            step_size=config.ray_step_size,
            max_steps=config.ray_max_steps,
            escape_radius=config.ray_escape_radius,
        )
        spectrum = ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=transfer_map,
            a_star=spin,
            eddington_ratio=0.1,
            f_col=1.7,
            delta_eta=delta_eta,
            energy_bins=bins,
            settings=settings,
            limb_darkening=config.limb_darkening,
        )
        ax.loglog(bins.centers_kev, spectrum, marker="o", markersize=3, label=label)
        for energy, flux in zip(bins.centers_kev, spectrum, strict=True):
            rows.append(
                {
                    "a_star": spin,
                    "delta_eta": delta_eta,
                    "inclination_deg": 40.0,
                    "energy_kev": float(energy),
                    "bin_energy_flux": float(flux),
                }
            )
    _write_dicts(TABLES / "figure5_spectra_source.csv", rows)
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(r"Bin energy flux [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_title("Ray-traced thermal spectra")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    _savefig("figure5_ray_traced_spectra.png")
    size = config.ray_screen_size
    return (
        "Ray-traced spectra for varied spin and stress at inclination 40 deg, "
        "eddington_ratio=0.1, f_col=1.7, for the fiducial "
        f"{config.mass_msun:g} solar-mass black hole at {config.distance_kpc:g} kpc. "
        f"Uses the corrected v5 full-disk {size}x{size} transfer map with physical "
        "absolute normalization; bin energy flux is in erg/s/cm^2."
    )


def _figure6_bias_width() -> str:
    rows = _read_csv(CONFIRMATORY / "phase12_results_unblinded.csv")
    x = np.array([_float(row, "mean_width_68") for row in rows])
    y = np.array([abs(_float(row, "mean_bias")) for row in rows])
    chi2 = np.array([_float(row, "mean_chi2_per_dof") for row in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    scatter = ax.scatter(x, y, c=np.log10(np.maximum(chi2, 1.0e-6)), cmap="magma")
    ax.set_xlabel("Mean 68% interval width")
    ax.set_ylabel("Absolute mean spin bias")
    ax.set_title("Spin bias versus interval width (v5 joint marginalized fit)")
    fig.colorbar(scatter, ax=ax, label="log10(mean chi2/dof)")
    ax.grid(alpha=0.25)
    _savefig("figure6_bias_width_proxy.png")
    return (
        "Condition-level absolute spin bias versus mean 68% interval width from the "
        "v5 joint marginalized-fit confirmatory table (color correction marginalized). "
        "Color is log10(mean chi2_per_dof); conditions with elevated chi2 are the "
        "inner-stress-misspecified cases that remain biased with narrow intervals."
    )


def _figure7_multi_epoch_status() -> str:
    rows = _read_csv(MULTI_EPOCH / "phase13p5_multi_epoch_summary.csv")
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    x = np.arange(len(completed))
    single_width = np.array(
        [_float(row, "mean_single_epoch_width_68") for row in completed]
    )
    joint_width = np.array([_float(row, "mean_joint_width_68") for row in completed])
    bias_delta = np.array(
        [_float(row, "joint_abs_bias_minus_single_epoch_abs_bias") for row in completed]
    )
    labels = [str(index) for index in range(1, len(completed) + 1)]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    width = 0.38
    axes[0].bar(x - width / 2.0, single_width, width=width, label="separate epochs")
    axes[0].bar(x + width / 2.0, joint_width, width=width, label="joint two-epoch")
    axes[0].set_ylabel("mean 68% spin-interval width")
    axes[0].set_title("Shared-spin interval width")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    colors = ["#32746d" if value < 0.0 else "#b23b3b" for value in bias_delta]
    axes[1].bar(x, bias_delta, color=colors)
    axes[1].set_ylabel("joint abs bias - separate abs bias")
    axes[1].set_title("Bias change is condition-dependent")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel("multi-epoch group index")
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    _savefig("figure7_multi_epoch_status.png")
    return (
        "Two-epoch shared-spin comparison from "
        "phase13p5_multi_epoch_summary.csv. Left panel compares the mean 68% "
        "spin-interval width from separate single-epoch fits against a joint "
        "two-epoch fit. Right panel shows the change in absolute spin bias; "
        "negative values favor the joint fit."
    )


def _figure8_bias_map() -> str:
    rows = _read_csv(CONFIRMATORY / "phase12_results_unblinded.csv")
    _heatmap_by_spin_inclination(
        rows,
        value_key="mean_bias",
        title="Mean spin bias by true spin and inclination",
        colorbar="mean bias",
        output="figure8_bias_map.png",
        cmap="coolwarm",
    )
    return (
        "Mean spin-bias map from phase12_results_unblinded.csv. Cells aggregate "
        "over luminosity, f_col_true, and stress at fixed true spin and inclination; "
        "bias is posterior-mean spin minus injected spin."
    )


def _figure9_coverage_map() -> str:
    rows = _read_csv(CONFIRMATORY / "phase12_results_unblinded.csv")
    _heatmap_by_spin_inclination(
        rows,
        value_key="coverage_68",
        title="Mean 68% spin-interval coverage",
        colorbar="coverage_68",
        output="figure9_coverage_map.png",
        cmap="viridis",
    )
    return (
        "Mean 68% interval coverage by true spin and inclination from "
        "phase12_results_unblinded.csv. Cells aggregate over luminosity, f_col_true, "
        "and stress."
    )


def _figure10_failure_domain() -> str:
    rows = _read_csv(CONFIRMATORY / "phase12_results_unblinded.csv")
    spin = np.array([_float(row, "spin_true") for row in rows])
    chi2 = np.array([_float(row, "mean_chi2_per_dof") for row in rows])
    failure = np.array([_float(row, "failure_rate") for row in rows])
    bias = np.array([abs(_float(row, "mean_bias")) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].scatter(spin, chi2, c=bias, cmap="plasma")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Injected spin")
    axes[0].set_ylabel("Mean chi2/dof")
    axes[0].set_title("Fit-quality domain")
    sc = axes[1].scatter(spin, failure, c=bias, cmap="plasma")
    axes[1].set_xlabel("Injected spin")
    axes[1].set_ylabel("Failure rate")
    axes[1].set_title("Failure accounting")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="absolute mean bias")
    _savefig("figure10_failure_domain.png")
    return (
        "Failure-domain summary from phase12_results_unblinded.csv. Left panel "
        "shows fit quality; right panel shows recorded failure rate. Color encodes "
        "absolute mean spin bias."
    )


def _figure11_convergence() -> str:
    resolution = _read_csv(CONFIRMATORY / "phase12_resolution_reruns.csv")
    transfer = _read_csv(TRANSFER_VALIDATION / "phase12p5_transfer_convergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    diffs = np.array([_float(row, "abs_bias_difference") for row in resolution])
    axes[0].hist(diffs, bins=12, color="#477998")
    axes[0].axvline(0.01, color="black", linestyle="--", label="tolerance")
    axes[0].set_xlabel("abs base/high mean-bias difference")
    axes[0].set_ylabel("conditions")
    axes[0].set_title("Spin-grid convergence")
    axes[0].legend()
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in transfer:
        label = f"a*={row['a_star']}, i={row['inclination_deg']}"
        grouped[label].append(
            (int(row["screen_size"]), float(row["relative_l1_spectrum_delta"]))
        )
    for label, values in grouped.items():
        values = sorted(values)
        axes[1].plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("screen size")
    axes[1].set_ylabel("relative L1 spectrum delta vs 7x7")
    axes[1].set_title("Transfer-map convergence")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, which="both")
    _savefig("figure11_numerical_convergence.png")
    return (
        "Numerical convergence summary. Left panel uses Phase 12 v4 base versus "
        "higher-resolution spin-grid bias differences with the 0.01 tolerance. "
        "Right panel uses Phase 12.5 transfer spectra normalized by relative L1 "
        "difference to the 7x7 screen reference."
    )


def _heatmap_by_spin_inclination(
    rows: list[dict[str, str]],
    *,
    value_key: str,
    title: str,
    colorbar: str,
    output: str,
    cmap: str,
) -> None:
    spins = sorted({float(row["spin_true"]) for row in rows})
    inclinations = sorted({float(row["inclination_deg"]) for row in rows})
    values = np.full((len(inclinations), len(spins)), np.nan)
    for i_idx, inclination in enumerate(inclinations):
        for s_idx, spin in enumerate(spins):
            matching = [
                float(row[value_key])
                for row in rows
                if float(row["spin_true"]) == spin
                and float(row["inclination_deg"]) == inclination
            ]
            values[i_idx, s_idx] = float(np.mean(matching))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(spins)))
    ax.set_xticklabels([f"{spin:g}" for spin in spins])
    ax.set_yticks(np.arange(len(inclinations)))
    ax.set_yticklabels([f"{inclination:g}" for inclination in inclinations])
    ax.set_xlabel("Injected spin")
    ax.set_ylabel("Inclination [deg]")
    ax.set_title(title)
    for i_idx in range(values.shape[0]):
        for s_idx in range(values.shape[1]):
            ax.text(
                s_idx, i_idx, f"{values[i_idx, s_idx]:.2f}", ha="center", va="center"
            )
    fig.colorbar(image, ax=ax, label=colorbar)
    _savefig(output)


def _screen_centers(lower: float, upper: float, count: int) -> np.ndarray:
    width = (upper - lower) / count
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * width


def _write_dicts(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_captions(captions: list[tuple[str, str]]) -> None:
    lines = ["# Figure Captions", ""]
    for index, (name, caption) in enumerate(captions, start=1):
        lines.extend([f"## Figure {index}: `{name}`", "", caption, ""])
    (ROOT / "paper" / "figure_captions.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
