"""Project constants.

Physical constants will be added only when the relevant equations are
registered and validated.
"""

from typing import Final

PACKAGE_NAME: Final[str] = "kerrdisk-uq"
SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0
GRAVITATIONAL_CONSTANT_SI: Final[float] = 6.67430e-11
PLANCK_CONSTANT_J_S: Final[float] = 6.626_070_15e-34
BOLTZMANN_CONSTANT_J_K: Final[float] = 1.380_649e-23
STEFAN_BOLTZMANN_CONSTANT_SI: Final[float] = 5.670_374_419e-8

# Astrophysical scale constants (SI unless noted).
SOLAR_MASS_KG: Final[float] = 1.988_409_9e30
PARSEC_M: Final[float] = 3.085_677_581_491_367_3e16
PROTON_MASS_KG: Final[float] = 1.672_621_923_69e-27
THOMSON_CROSS_SECTION_M2: Final[float] = 6.652_458_732_1e-29
ELECTRON_VOLT_J: Final[float] = 1.602_176_634e-19
KILO_ELECTRON_VOLT_J: Final[float] = 1.602_176_634e-16
# 1 W/m^2 = 1e3 erg/s/cm^2.
WATT_PER_M2_TO_ERG_PER_S_CM2: Final[float] = 1.0e3
