"""
Stellar parameter management and validation.
"""

from typing import Optional, Dict, Any
import numpy as np


class StellarParameters:
    """
    Container for stellar physical parameters.

    Attributes
    ----------
    teff : float
        Effective temperature in Kelvin
    logg : float
        Surface gravity log g (cgs)
    feh : float
        Metallicity [Fe/H]
    alpha_fe : float
        Alpha enhancement [α/Fe]
    """

    def __init__(
        self,
        teff: Optional[float] = None,
        logg: Optional[float] = None,
        feh: Optional[float] = None,
        alpha_fe: Optional[float] = None,
        uncertainties: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize stellar parameters.

        Parameters
        ----------
        teff : float, optional
            Effective temperature in K
        logg : float, optional
            Surface gravity (log g)
        feh : float, optional
            Metallicity [Fe/H]
        alpha_fe : float, optional
            Alpha enhancement [α/Fe]
        uncertainties : dict, optional
            Dictionary of parameter uncertainties
        """
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.alpha_fe = alpha_fe

        self.uncertainties = uncertainties or {}

    def validate(self) -> bool:
        """
        Validate that parameters are within physically reasonable ranges.

        Returns
        -------
        bool
            True if parameters are valid
        """
        valid = True

        if self.teff is not None:
            if not (2000 <= self.teff <= 50000):
                print(f"Warning: Teff={self.teff} K is outside typical range [2000, 50000] K")
                valid = False

        if self.logg is not None:
            if not (-1.0 <= self.logg <= 6.0):
                print(f"Warning: log g={self.logg} is outside typical range [-1, 6]")
                valid = False

        if self.feh is not None:
            if not (-5.0 <= self.feh <= 1.0):
                print(f"Warning: [Fe/H]={self.feh} is outside typical range [-5, 1]")
                valid = False

        if self.alpha_fe is not None:
            if not (-0.5 <= self.alpha_fe <= 1.0):
                print(f"Warning: [α/Fe]={self.alpha_fe} is outside typical range [-0.5, 1]")
                valid = False

        return valid

    def is_complete(self) -> bool:
        """
        Check if all main parameters are defined.

        Returns
        -------
        bool
            True if Teff, log g, and [Fe/H] are all defined
        """
        return self.teff is not None and self.logg is not None and self.feh is not None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary.

        Returns
        -------
        dict
            Dictionary of parameters
        """
        return {
            "teff": self.teff,
            "logg": self.logg,
            "feh": self.feh,
            "alpha_fe": self.alpha_fe,
            "uncertainties": self.uncertainties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StellarParameters":
        """
        Create StellarParameters from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with parameter values

        Returns
        -------
        StellarParameters
            New parameters object
        """
        return cls(
            teff=data.get("teff"),
            logg=data.get("logg"),
            feh=data.get("feh"),
            alpha_fe=data.get("alpha_fe"),
            uncertainties=data.get("uncertainties"),
        )

    def spectral_type_estimate(self) -> Optional[str]:
        """
        Estimate spectral type from Teff.

        Returns
        -------
        str or None
            Estimated spectral type (e.g., 'G2V')
        """
        if self.teff is None:
            return None

        # Simplified spectral type mapping
        if self.teff >= 30000:
            spec_type = "O"
        elif self.teff >= 10000:
            spec_type = "B"
        elif self.teff >= 7500:
            spec_type = "A"
        elif self.teff >= 6000:
            spec_type = "F"
        elif self.teff >= 5200:
            spec_type = "G"
        elif self.teff >= 3700:
            spec_type = "K"
        else:
            spec_type = "M"

        # Luminosity class from log g
        if self.logg is not None:
            if self.logg < 1.0:
                lum_class = "I"
            elif self.logg < 2.5:
                lum_class = "III"
            elif self.logg < 3.5:
                lum_class = "IV"
            else:
                lum_class = "V"
        else:
            lum_class = ""

        return f"{spec_type}{lum_class}"

    def __repr__(self):
        parts = []
        if self.teff is not None:
            parts.append(f"Teff={self.teff:.0f}K")
        if self.logg is not None:
            parts.append(f"log g={self.logg:.2f}")
        if self.feh is not None:
            parts.append(f"[Fe/H]={self.feh:.2f}")
        if self.alpha_fe is not None:
            parts.append(f"[α/Fe]={self.alpha_fe:.2f}")

        return f"StellarParameters({', '.join(parts)})"


class ParameterGrid:
    """
    Grid of stellar parameters for library organization.
    """

    def __init__(
        self,
        teff_range: tuple = (3000, 10000),
        teff_step: float = 250,
        logg_range: tuple = (0.0, 5.0),
        logg_step: float = 0.5,
        feh_range: tuple = (-2.5, 0.5),
        feh_step: float = 0.5,
    ):
        """
        Initialize parameter grid.

        Parameters
        ----------
        teff_range : tuple
            (min, max) Teff in K
        teff_step : float
            Teff grid step
        logg_range : tuple
            (min, max) log g
        logg_step : float
            log g grid step
        feh_range : tuple
            (min, max) [Fe/H]
        feh_step : float
            [Fe/H] grid step
        """
        self.teff_grid = np.arange(teff_range[0], teff_range[1] + teff_step, teff_step)
        self.logg_grid = np.arange(logg_range[0], logg_range[1] + logg_step, logg_step)
        self.feh_grid = np.arange(feh_range[0], feh_range[1] + feh_step, feh_step)

    def find_nearest_grid_point(self, teff: float, logg: float, feh: float) -> Dict[str, float]:
        """
        Find nearest grid point to given parameters.

        Parameters
        ----------
        teff : float
            Effective temperature
        logg : float
            Surface gravity
        feh : float
            Metallicity

        Returns
        -------
        dict
            Dictionary with nearest grid point parameters
        """
        nearest_teff = self.teff_grid[np.argmin(np.abs(self.teff_grid - teff))]
        nearest_logg = self.logg_grid[np.argmin(np.abs(self.logg_grid - logg))]
        nearest_feh = self.feh_grid[np.argmin(np.abs(self.feh_grid - feh))]

        return {
            "teff": nearest_teff,
            "logg": nearest_logg,
            "feh": nearest_feh,
        }

    def get_grid_size(self) -> int:
        """
        Get total number of grid points.

        Returns
        -------
        int
            Total number of grid points
        """
        return len(self.teff_grid) * len(self.logg_grid) * len(self.feh_grid)

    def __repr__(self):
        return (
            f"ParameterGrid(Teff: {len(self.teff_grid)} points, "
            f"log g: {len(self.logg_grid)} points, "
            f"[Fe/H]: {len(self.feh_grid)} points, "
            f"total: {self.get_grid_size()} points)"
        )
