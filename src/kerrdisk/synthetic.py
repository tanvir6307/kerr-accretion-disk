"""Deterministic synthetic spectra and noise models."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type MetadataValue = str | int | float | bool | None
type Metadata = dict[str, MetadataValue]
type NoiseModelName = Literal["gaussian_relative", "poisson_counts"]
type PhotonFluxDensity = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class EnergyBins:
    """Detector-independent energy-bin grid in keV."""

    edges_kev: FloatArray
    centers_kev: FloatArray
    widths_kev: FloatArray


@dataclass(frozen=True)
class SeedManifest:
    """Hierarchical deterministic seeds used for one synthetic epoch."""

    master_seed: int
    condition_label: str
    condition_seed: int
    epoch_index: int
    epoch_seed: int
    noise_seed: int


@dataclass(frozen=True)
class GaussianRelativeNoise:
    """Gaussian debug-noise model with fixed fractional error."""

    relative_error: float


@dataclass(frozen=True)
class PoissonNoise:
    """Scalar effective-area and exposure model for count noise."""

    effective_area_cm2: float
    exposure_s: float


@dataclass(frozen=True)
class SyntheticEpoch:
    """One synthetic detector-independent spectrum."""

    epoch_index: int
    energy_bin_edges_kev: FloatArray
    energy_bin_centers_kev: FloatArray
    photon_flux: FloatArray
    energy_flux: FloatArray
    expectation: FloatArray
    observed: FloatArray
    variance: FloatArray
    noise_model: NoiseModelName
    seed_manifest: SeedManifest
    truth_metadata: Metadata


@dataclass(frozen=True)
class MultiEpochDataset:
    """Synthetic spectra and shared truth metadata for multiple epochs."""

    epochs: tuple[SyntheticEpoch, ...]
    truth_metadata: Metadata


def _validate_nonnegative_seed(master_seed: int) -> None:
    if master_seed < 0:
        msg = "master_seed must be nonnegative"
        raise ValueError(msg)


def _as_positive_1d(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(array <= 0.0):
        msg = f"{name} must contain only positive values"
        raise ValueError(msg)
    return array


def _as_nonnegative_1d(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(array < 0.0):
        msg = f"{name} must contain only nonnegative values"
        raise ValueError(msg)
    return array


def _validate_metadata(metadata: Metadata | None) -> Metadata:
    if metadata is None:
        return {}
    return dict(metadata)


def derive_seed(master_seed: int, *labels: str | int) -> int:
    """Return a deterministic 32-bit seed from a master seed and labels."""

    _validate_nonnegative_seed(master_seed)
    digest = blake2b(digest_size=8)
    digest.update(str(master_seed).encode("utf-8"))
    for label in labels:
        digest.update(b"\0")
        digest.update(str(label).encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big") % (2**32)


def make_seed_manifest(
    master_seed: int,
    *,
    condition_label: str,
    epoch_index: int,
) -> SeedManifest:
    """Return the hierarchical seeds for one condition and epoch."""

    _validate_nonnegative_seed(master_seed)
    if epoch_index < 0:
        msg = "epoch_index must be nonnegative"
        raise ValueError(msg)
    condition_seed = derive_seed(master_seed, "condition", condition_label)
    epoch_seed = derive_seed(condition_seed, "epoch", epoch_index)
    noise_seed = derive_seed(epoch_seed, "noise")
    return SeedManifest(
        master_seed=master_seed,
        condition_label=condition_label,
        condition_seed=condition_seed,
        epoch_index=epoch_index,
        epoch_seed=epoch_seed,
        noise_seed=noise_seed,
    )


def make_log_energy_bins(
    *,
    energy_min_kev: float = 0.1,
    energy_max_kev: float = 20.0,
    bin_count: int = 120,
) -> EnergyBins:
    """Return logarithmic detector-independent energy bins."""

    if not isfinite(energy_min_kev) or energy_min_kev <= 0.0:
        msg = "energy_min_kev must be finite and positive"
        raise ValueError(msg)
    if not isfinite(energy_max_kev) or energy_max_kev <= energy_min_kev:
        msg = "energy_max_kev must be finite and larger than energy_min_kev"
        raise ValueError(msg)
    if bin_count < 1:
        msg = "bin_count must be positive"
        raise ValueError(msg)
    edges = np.geomspace(energy_min_kev, energy_max_kev, bin_count + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    return EnergyBins(
        edges_kev=edges.astype(np.float64),
        centers_kev=centers.astype(np.float64),
        widths_kev=widths.astype(np.float64),
    )


def validate_energy_bins(energy_bins: EnergyBins) -> None:
    """Validate an energy-bin object."""

    edges = _as_positive_1d("energy_bin_edges_kev", energy_bins.edges_kev)
    centers = _as_positive_1d("energy_bin_centers_kev", energy_bins.centers_kev)
    widths = _as_positive_1d("energy_bin_widths_kev", energy_bins.widths_kev)
    if edges.size != centers.size + 1 or centers.shape != widths.shape:
        msg = "energy-bin edge, center, and width shapes are inconsistent"
        raise ValueError(msg)
    if not np.all(np.diff(edges) > 0.0):
        msg = "energy_bin_edges_kev must be strictly increasing"
        raise ValueError(msg)


def integrated_photon_flux_from_density(
    energy_bins: EnergyBins,
    photon_flux_density: PhotonFluxDensity,
) -> FloatArray:
    """Evaluate bin-integrated photon flux with the midpoint rule."""

    validate_energy_bins(energy_bins)
    density = np.asarray(photon_flux_density(energy_bins.centers_kev), dtype=np.float64)
    if density.shape != energy_bins.centers_kev.shape:
        msg = "photon_flux_density must return one value per energy-bin center"
        raise ValueError(msg)
    density = _as_nonnegative_1d("photon_flux_density", density)
    return density * energy_bins.widths_kev


def _energy_flux_from_photon_flux(
    energy_bins: EnergyBins,
    photon_flux: ArrayLike,
) -> FloatArray:
    validate_energy_bins(energy_bins)
    photon_flux_array = _as_nonnegative_1d("photon_flux", photon_flux)
    if photon_flux_array.shape != energy_bins.centers_kev.shape:
        msg = "photon_flux must have one value per energy bin"
        raise ValueError(msg)
    return energy_bins.centers_kev * photon_flux_array


def _validate_gaussian_noise(noise: GaussianRelativeNoise) -> None:
    if not isfinite(noise.relative_error) or noise.relative_error <= 0.0:
        msg = "relative_error must be finite and positive"
        raise ValueError(msg)


def _validate_poisson_noise(noise: PoissonNoise) -> None:
    if not isfinite(noise.effective_area_cm2) or noise.effective_area_cm2 <= 0.0:
        msg = "effective_area_cm2 must be finite and positive"
        raise ValueError(msg)
    if not isfinite(noise.exposure_s) or noise.exposure_s <= 0.0:
        msg = "exposure_s must be finite and positive"
        raise ValueError(msg)


def generate_gaussian_epoch(
    energy_bins: EnergyBins,
    photon_flux: ArrayLike,
    noise: GaussianRelativeNoise,
    *,
    master_seed: int,
    condition_label: str,
    epoch_index: int,
    truth_metadata: Metadata | None = None,
) -> SyntheticEpoch:
    """Generate one Gaussian relative-error synthetic epoch."""

    _validate_gaussian_noise(noise)
    energy_flux = _energy_flux_from_photon_flux(energy_bins, photon_flux)
    if np.any(energy_flux <= 0.0):
        msg = "energy_flux must be positive for Gaussian relative-error noise"
        raise ValueError(msg)
    manifest = make_seed_manifest(
        master_seed,
        condition_label=condition_label,
        epoch_index=epoch_index,
    )
    rng = np.random.default_rng(manifest.noise_seed)
    sigma = noise.relative_error * energy_flux
    observed = rng.normal(loc=energy_flux, scale=sigma)
    return SyntheticEpoch(
        epoch_index=epoch_index,
        energy_bin_edges_kev=energy_bins.edges_kev.copy(),
        energy_bin_centers_kev=energy_bins.centers_kev.copy(),
        photon_flux=np.asarray(photon_flux, dtype=np.float64).copy(),
        energy_flux=energy_flux,
        expectation=energy_flux.copy(),
        observed=np.asarray(observed, dtype=np.float64),
        variance=sigma * sigma,
        noise_model="gaussian_relative",
        seed_manifest=manifest,
        truth_metadata=_validate_metadata(truth_metadata),
    )


def generate_poisson_epoch(
    energy_bins: EnergyBins,
    photon_flux: ArrayLike,
    noise: PoissonNoise,
    *,
    master_seed: int,
    condition_label: str,
    epoch_index: int,
    truth_metadata: Metadata | None = None,
) -> SyntheticEpoch:
    """Generate one Poisson-count synthetic epoch."""

    _validate_poisson_noise(noise)
    photon_flux_array = _as_nonnegative_1d("photon_flux", photon_flux)
    energy_flux = _energy_flux_from_photon_flux(energy_bins, photon_flux_array)
    expected_counts = photon_flux_array * noise.effective_area_cm2 * noise.exposure_s
    if np.any(expected_counts <= 0.0):
        msg = "expected_counts must be positive for Poisson noise"
        raise ValueError(msg)
    manifest = make_seed_manifest(
        master_seed,
        condition_label=condition_label,
        epoch_index=epoch_index,
    )
    rng = np.random.default_rng(manifest.noise_seed)
    observed = rng.poisson(expected_counts).astype(np.float64)
    return SyntheticEpoch(
        epoch_index=epoch_index,
        energy_bin_edges_kev=energy_bins.edges_kev.copy(),
        energy_bin_centers_kev=energy_bins.centers_kev.copy(),
        photon_flux=photon_flux_array.copy(),
        energy_flux=energy_flux,
        expectation=expected_counts,
        observed=observed,
        variance=expected_counts.copy(),
        noise_model="poisson_counts",
        seed_manifest=manifest,
        truth_metadata=_validate_metadata(truth_metadata),
    )


def generate_multi_epoch_dataset(
    energy_bins: EnergyBins,
    photon_flux_by_epoch: Sequence[ArrayLike],
    noise: GaussianRelativeNoise | PoissonNoise,
    *,
    master_seed: int,
    condition_label: str,
    truth_metadata: Metadata | None = None,
    epoch_truth_metadata: Sequence[Metadata] | None = None,
) -> MultiEpochDataset:
    """Generate a deterministic multi-epoch synthetic dataset."""

    if not photon_flux_by_epoch:
        msg = "photon_flux_by_epoch must contain at least one epoch"
        raise ValueError(msg)
    if epoch_truth_metadata is not None and len(epoch_truth_metadata) != len(
        photon_flux_by_epoch
    ):
        msg = "epoch_truth_metadata length must match photon_flux_by_epoch"
        raise ValueError(msg)

    epochs: list[SyntheticEpoch] = []
    for epoch_index, photon_flux in enumerate(photon_flux_by_epoch):
        metadata = (
            None if epoch_truth_metadata is None else epoch_truth_metadata[epoch_index]
        )
        if isinstance(noise, GaussianRelativeNoise):
            epoch = generate_gaussian_epoch(
                energy_bins,
                photon_flux,
                noise,
                master_seed=master_seed,
                condition_label=condition_label,
                epoch_index=epoch_index,
                truth_metadata=metadata,
            )
        else:
            epoch = generate_poisson_epoch(
                energy_bins,
                photon_flux,
                noise,
                master_seed=master_seed,
                condition_label=condition_label,
                epoch_index=epoch_index,
                truth_metadata=metadata,
            )
        epochs.append(epoch)

    return MultiEpochDataset(
        epochs=tuple(epochs),
        truth_metadata=_validate_metadata(truth_metadata),
    )
