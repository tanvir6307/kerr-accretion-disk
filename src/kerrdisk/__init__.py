"""KerrDisk-UQ package scaffolding."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kerrdisk-uq")
except PackageNotFoundError:
    __version__ = "0.0.0+editable"

__all__ = ["__version__"]
