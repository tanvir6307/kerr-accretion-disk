"""Smoke tests for bootstrap imports."""

import importlib


def test_public_modules_import() -> None:
    modules = [
        "kerrdisk",
        "kerrdisk.atmosphere",
        "kerrdisk.circular_orbits",
        "kerrdisk.cli",
        "kerrdisk.config",
        "kerrdisk.confirmatory",
        "kerrdisk.constants",
        "kerrdisk.diagnostics",
        "kerrdisk.disk_flux",
        "kerrdisk.geodesics",
        "kerrdisk.inference",
        "kerrdisk.io",
        "kerrdisk.isco",
        "kerrdisk.likelihood",
        "kerrdisk.metric",
        "kerrdisk.plotting",
        "kerrdisk.raytrace",
        "kerrdisk.screening",
        "kerrdisk.spectrum",
        "kerrdisk.synthetic",
        "kerrdisk.thermal_spectrum",
        "kerrdisk.units",
    ]

    for module in modules:
        importlib.import_module(module)
