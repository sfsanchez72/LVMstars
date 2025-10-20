"""
LVMstars: LVM Stellar Library

A package for building a stellar library from LVM observations,
with GAIA cross-matching, sky/ISM subtraction, dust correction,
and stellar parameter tagging.
"""

__version__ = "0.1.0"

from .spectrum import StellarSpectrum
from .gaia import GaiaCrossmatcher, GaiaMatch
from .subtraction import SkySubtractor
from .dust import DustCorrector
from .parameters import StellarParameters

__all__ = [
    "StellarSpectrum",
    "GaiaCrossmatcher",
    "GaiaMatch",
    "SkySubtractor",
    "DustCorrector",
    "StellarParameters",
]
