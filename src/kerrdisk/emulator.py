"""Precomputed interpolated thin-disk spectral grid.

A ``SpectralGrid`` evaluates a spectral forward model once on a regular grid of
parameters and interpolates between the nodes. This makes an expensive backend
(such as the ray-traced transfer calculation) affordable inside a sampler: the
costly transfer maps are computed only at the grid nodes, and every posterior
evaluation is a fast multilinear interpolation.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

type FloatArray = NDArray[np.float64]
type GridModel = Callable[[Mapping[str, float]], ArrayLike]


@dataclass
class SpectralGrid:
    """Regular parameter grid of binned spectra with multilinear interpolation."""

    axis_names: tuple[str, ...]
    axis_values: tuple[FloatArray, ...]
    energy_centers_kev: FloatArray
    spectra: FloatArray
    _interpolator: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.axis_names) != len(self.axis_values):
            msg = "axis_names and axis_values must have the same length"
            raise ValueError(msg)
        if not self.axis_names:
            msg = "at least one axis is required"
            raise ValueError(msg)
        expected_shape = tuple(values.size for values in self.axis_values)
        if self.spectra.shape != (*expected_shape, self.energy_centers_kev.size):
            msg = "spectra shape must match the axis grid and energy bins"
            raise ValueError(msg)
        for name, values in zip(self.axis_names, self.axis_values, strict=True):
            if values.ndim != 1 or values.size < 2:
                msg = f"axis {name!r} must have at least two nodes"
                raise ValueError(msg)
            if not np.all(np.diff(values) > 0.0):
                msg = f"axis {name!r} must be strictly increasing"
                raise ValueError(msg)
        if not np.all(np.isfinite(self.spectra)):
            msg = "spectra must contain only finite values"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "_interpolator",
            RegularGridInterpolator(
                self.axis_values,
                self.spectra,
                method="linear",
                bounds_error=True,
            ),
        )

    @property
    def dimension(self) -> int:
        """Return the number of grid axes."""

        return len(self.axis_names)

    def evaluate(self, point: ArrayLike) -> FloatArray:
        """Interpolate the spectrum at a point given in axis order."""

        values = np.asarray(point, dtype=np.float64)
        if values.shape != (self.dimension,):
            msg = f"point must have shape ({self.dimension},)"
            raise ValueError(msg)
        result = self._interpolator(values.reshape(1, self.dimension))
        return np.asarray(result, dtype=np.float64).reshape(-1)

    def evaluate_named(self, **named: float) -> FloatArray:
        """Interpolate the spectrum from named axis coordinates."""

        missing = set(self.axis_names) - set(named)
        if missing:
            msg = f"missing axis coordinates: {sorted(missing)}"
            raise ValueError(msg)
        point = np.array([named[name] for name in self.axis_names], dtype=np.float64)
        return self.evaluate(point)

    def save(self, path: Path) -> None:
        """Write the grid to a compressed NumPy archive."""

        path.parent.mkdir(parents=True, exist_ok=True)
        axis_lengths = np.array(
            [values.size for values in self.axis_values], dtype=np.int64
        )
        axis_concatenated = np.concatenate(self.axis_values)
        np.savez_compressed(
            path,
            axis_names=np.array(json.dumps(list(self.axis_names)), dtype=np.str_),
            axis_lengths=axis_lengths,
            axis_concatenated=axis_concatenated,
            energy_centers_kev=self.energy_centers_kev,
            spectra=self.spectra,
        )


def build_spectral_grid(
    *,
    axes: Mapping[str, ArrayLike],
    energy_centers_kev: ArrayLike,
    model: GridModel,
) -> SpectralGrid:
    """Evaluate ``model`` on the Cartesian product of ``axes`` and store it."""

    if not axes:
        msg = "at least one axis is required"
        raise ValueError(msg)
    axis_names = tuple(axes)
    axis_values = tuple(np.asarray(axes[name], dtype=np.float64) for name in axis_names)
    energy = np.asarray(energy_centers_kev, dtype=np.float64)
    if energy.ndim != 1 or energy.size < 1:
        msg = "energy_centers_kev must be a nonempty one-dimensional array"
        raise ValueError(msg)

    shape = tuple(values.size for values in axis_values)
    spectra = np.empty((*shape, energy.size), dtype=np.float64)
    for index in np.ndindex(*shape):
        point = {
            name: float(axis_values[axis][index[axis]])
            for axis, name in enumerate(axis_names)
        }
        spectrum = np.asarray(model(point), dtype=np.float64)
        if spectrum.shape != (energy.size,):
            msg = "model must return a spectrum matching energy_centers_kev"
            raise ValueError(msg)
        spectra[index] = spectrum

    return SpectralGrid(
        axis_names=axis_names,
        axis_values=axis_values,
        energy_centers_kev=energy,
        spectra=spectra,
    )


def load_spectral_grid(path: Path) -> SpectralGrid:
    """Load a spectral grid written by :meth:`SpectralGrid.save`."""

    with np.load(path) as archive:
        axis_names = tuple(json.loads(str(archive["axis_names"])))
        lengths = np.asarray(archive["axis_lengths"], dtype=np.int64)
        concatenated = np.asarray(archive["axis_concatenated"], dtype=np.float64)
        boundaries = np.cumsum(lengths)[:-1]
        axis_values = tuple(
            np.asarray(values, dtype=np.float64)
            for values in np.split(concatenated, boundaries)
        )
        return SpectralGrid(
            axis_names=axis_names,
            axis_values=axis_values,
            energy_centers_kev=np.asarray(
                archive["energy_centers_kev"], dtype=np.float64
            ),
            spectra=np.asarray(archive["spectra"], dtype=np.float64),
        )
