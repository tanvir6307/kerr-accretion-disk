"""Physical scale conversions between geometric units and SI/observed units.

These helpers convert a black-hole mass, distance, and Eddington ratio into the
SI quantities required to normalize a Kerr thin-disk spectrum. They isolate the
dimensional boundary from the dimensionless geometric-unit physics.
"""

from math import isfinite

from kerrdisk.constants import (
    GRAVITATIONAL_CONSTANT_SI,
    KILO_ELECTRON_VOLT_J,
    PARSEC_M,
    PLANCK_CONSTANT_J_S,
    PROTON_MASS_KG,
    SOLAR_MASS_KG,
    SPEED_OF_LIGHT_M_PER_S,
    THOMSON_CROSS_SECTION_M2,
)


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)


def black_hole_mass_kg(mass_msun: float) -> float:
    """Return the black-hole mass in kilograms."""

    _require_positive("mass_msun", mass_msun)
    return mass_msun * SOLAR_MASS_KG


def gravitational_radius_m(mass_kg: float) -> float:
    """Return the gravitational radius `r_g = G M / c^2` in meters."""

    _require_positive("mass_kg", mass_kg)
    return GRAVITATIONAL_CONSTANT_SI * mass_kg / SPEED_OF_LIGHT_M_PER_S**2


def distance_m(distance_kpc: float) -> float:
    """Return a distance in meters from kiloparsecs."""

    _require_positive("distance_kpc", distance_kpc)
    return distance_kpc * 1.0e3 * PARSEC_M


def eddington_luminosity_w(mass_kg: float) -> float:
    """Return the Eddington luminosity `4 pi G M m_p c / sigma_T` in watts."""

    _require_positive("mass_kg", mass_kg)
    return (
        4.0
        * 3.141592653589793
        * GRAVITATIONAL_CONSTANT_SI
        * mass_kg
        * PROTON_MASS_KG
        * SPEED_OF_LIGHT_M_PER_S
        / THOMSON_CROSS_SECTION_M2
    )


def accretion_rate_kg_s(
    mass_kg: float,
    eddington_ratio: float,
    efficiency: float,
) -> float:
    """Return the accretion rate `Mdot = ell L_Edd / (eta c^2)` in kg/s.

    ``eddington_ratio`` is the bolometric ``L / L_Edd`` and ``efficiency`` is the
    radiative efficiency ``eta`` that links luminosity and accretion rate through
    ``L = eta Mdot c^2``.
    """

    _require_positive("mass_kg", mass_kg)
    _require_positive("eddington_ratio", eddington_ratio)
    _require_positive("efficiency", efficiency)
    luminosity = eddington_ratio * eddington_luminosity_w(mass_kg)
    return luminosity / (efficiency * SPEED_OF_LIGHT_M_PER_S**2)


def frequency_hz_from_kev(energy_kev: float) -> float:
    """Return the photon frequency in hertz for a photon energy in keV."""

    _require_positive("energy_kev", energy_kev)
    return energy_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S


def observer_distance_rg(distance_kpc: float, mass_msun: float) -> float:
    """Return the astronomical distance expressed in gravitational radii.

    The image-plane solid angle used by the transfer map is
    ``d_alpha d_beta / observer_distance^2`` with screen coordinates in
    gravitational radii, so passing this value as ``observer_distance`` makes the
    stored solid angle the true physical observer solid angle in steradians.
    """

    return distance_m(distance_kpc) / gravitational_radius_m(
        black_hole_mass_kg(mass_msun)
    )
