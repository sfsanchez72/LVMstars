"""
Dust correction using DAP (Data Analysis Pipeline) results.
"""

import numpy as np
from typing import Optional, Dict, Any
from .spectrum import StellarSpectrum
import warnings


class DustCorrector:
    """
    Apply dust extinction corrections to stellar spectra using DAP results.
    """

    def __init__(self, extinction_law: str = "ccm89"):
        """
        Initialize the DustCorrector.

        Parameters
        ----------
        extinction_law : str
            Extinction law to use ('ccm89', 'odonnell94', 'calzetti00', 'fitzpatrick99')
        """
        self.extinction_law = extinction_law

    def correct(
        self,
        spectrum: StellarSpectrum,
        dap_results: Dict[str, Any],
        ebv: Optional[float] = None,
        rv: float = 3.1,
    ) -> StellarSpectrum:
        """
        Apply dust correction to a spectrum.

        Parameters
        ----------
        spectrum : StellarSpectrum
            Input spectrum
        dap_results : dict
            DAP results containing extinction information
            Expected keys: 'ebv' (color excess E(B-V))
        ebv : float, optional
            E(B-V) color excess (overrides dap_results if provided)
        rv : float
            Total to selective extinction ratio (default: 3.1)

        Returns
        -------
        StellarSpectrum
            Dust-corrected spectrum
        """
        # Get E(B-V) from parameters or DAP results
        if ebv is None:
            ebv = dap_results.get("ebv")
            if ebv is None:
                warnings.warn("No E(B-V) value found, skipping dust correction")
                return spectrum

        if ebv <= 0:
            # No dust correction needed
            return spectrum

        # Calculate extinction curve
        extinction = self._calculate_extinction(spectrum.wavelength, ebv, rv)

        # Apply correction: F_corrected = F_observed * 10^(0.4 * A_lambda)
        correction_factor = 10.0 ** (0.4 * extinction)
        corrected_flux = spectrum.flux * correction_factor
        corrected_error = spectrum.flux_error * correction_factor

        # Create corrected spectrum
        corrected_spectrum = StellarSpectrum(
            wavelength=spectrum.wavelength.copy(),
            flux=corrected_flux,
            flux_error=corrected_error,
            coordinates=spectrum.coordinates,
            metadata=spectrum.metadata.copy(),
        )

        # Copy stellar parameters
        corrected_spectrum.parameters = spectrum.parameters.copy()

        # Add correction metadata
        corrected_spectrum.metadata["dust_correction"] = {
            "ebv": ebv,
            "rv": rv,
            "extinction_law": self.extinction_law,
        }

        return corrected_spectrum

    def _calculate_extinction(
        self,
        wavelength: np.ndarray,
        ebv: float,
        rv: float,
    ) -> np.ndarray:
        """
        Calculate extinction A_lambda as a function of wavelength.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in Angstroms
        ebv : float
            E(B-V) color excess
        rv : float
            Total to selective extinction ratio

        Returns
        -------
        np.ndarray
            Extinction A_lambda at each wavelength
        """
        if self.extinction_law == "ccm89":
            return self._ccm89_extinction(wavelength, ebv, rv)
        elif self.extinction_law == "odonnell94":
            return self._odonnell94_extinction(wavelength, ebv, rv)
        elif self.extinction_law == "calzetti00":
            return self._calzetti00_extinction(wavelength, ebv, rv)
        elif self.extinction_law == "fitzpatrick99":
            return self._fitzpatrick99_extinction(wavelength, ebv, rv)
        else:
            raise ValueError(f"Unknown extinction law: {self.extinction_law}")

    def _ccm89_extinction(
        self,
        wavelength: np.ndarray,
        ebv: float,
        rv: float,
    ) -> np.ndarray:
        """
        CCM 1989 extinction law (Cardelli, Clayton, & Mathis 1989).

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength in Angstroms
        ebv : float
            E(B-V)
        rv : float
            R_V

        Returns
        -------
        np.ndarray
            A_lambda
        """
        # Convert wavelength to inverse microns
        x = 10000.0 / wavelength  # 1/microns

        a = np.zeros_like(x)
        b = np.zeros_like(x)

        # Infrared (0.3 - 1.1 microns^-1)
        ir_mask = (x >= 0.3) & (x < 1.1)
        if np.any(ir_mask):
            a[ir_mask] = 0.574 * x[ir_mask] ** 1.61
            b[ir_mask] = -0.527 * x[ir_mask] ** 1.61

        # Optical/NIR (1.1 - 3.3 microns^-1)
        opt_mask = (x >= 1.1) & (x < 3.3)
        if np.any(opt_mask):
            y = x[opt_mask] - 1.82
            a[opt_mask] = (
                1
                + 0.17699 * y
                - 0.50447 * y**2
                - 0.02427 * y**3
                + 0.72085 * y**4
                + 0.01979 * y**5
                - 0.77530 * y**6
                + 0.32999 * y**7
            )
            b[opt_mask] = (
                1.41338 * y
                + 2.28305 * y**2
                + 1.07233 * y**3
                - 5.38434 * y**4
                - 0.62251 * y**5
                + 5.30260 * y**6
                - 2.09002 * y**7
            )

        # UV (3.3 - 8.0 microns^-1)
        uv_mask = (x >= 3.3) & (x < 8.0)
        if np.any(uv_mask):
            fa = np.zeros_like(x[uv_mask])
            fb = np.zeros_like(x[uv_mask])

            fuv_mask = x[uv_mask] >= 5.9
            if np.any(fuv_mask):
                y_uv = x[uv_mask][fuv_mask] - 5.9
                fa[fuv_mask] = -0.04473 * y_uv**2 - 0.009779 * y_uv**3
                fb[fuv_mask] = 0.2130 * y_uv**2 + 0.1207 * y_uv**3

            a[uv_mask] = (
                1.752 - 0.316 * x[uv_mask] - 0.104 / ((x[uv_mask] - 4.67) ** 2 + 0.341) + fa
            )
            b[uv_mask] = (
                -3.090 + 1.825 * x[uv_mask] + 1.206 / ((x[uv_mask] - 4.62) ** 2 + 0.263) + fb
            )

        # Calculate A_lambda / A_V
        alambda_av = a + b / rv

        # Calculate A_lambda = A_V * (A_lambda / A_V) = E(B-V) * R_V * (A_lambda / A_V)
        av = rv * ebv
        alambda = av * alambda_av

        return alambda

    def _odonnell94_extinction(
        self,
        wavelength: np.ndarray,
        ebv: float,
        rv: float,
    ) -> np.ndarray:
        """
        O'Donnell 1994 extinction law (update to CCM89).
        Similar to CCM89 but with updated coefficients in optical.
        """
        # For simplicity, use CCM89 (O'Donnell has minor differences)
        return self._ccm89_extinction(wavelength, ebv, rv)

    def _calzetti00_extinction(
        self,
        wavelength: np.ndarray,
        ebv: float,
        rv: float = 4.05,
    ) -> np.ndarray:
        """
        Calzetti et al. 2000 starburst attenuation law.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength in Angstroms
        ebv : float
            E(B-V)
        rv : float
            R_V (default 4.05 for Calzetti law)

        Returns
        -------
        np.ndarray
            A_lambda
        """
        wave_micron = wavelength / 10000.0

        k = np.zeros_like(wave_micron)

        # 0.12 - 0.63 microns
        mask1 = (wave_micron >= 0.12) & (wave_micron < 0.63)
        k[mask1] = (
            2.659
            * (
                -2.156
                + 1.509 / wave_micron[mask1]
                - 0.198 / wave_micron[mask1] ** 2
                + 0.011 / wave_micron[mask1] ** 3
            )
            + rv
        )

        # 0.63 - 2.2 microns
        mask2 = (wave_micron >= 0.63) & (wave_micron <= 2.2)
        k[mask2] = 2.659 * (-1.857 + 1.040 / wave_micron[mask2]) + rv

        # Calculate A_lambda
        alambda = k * ebv

        return alambda

    def _fitzpatrick99_extinction(
        self,
        wavelength: np.ndarray,
        ebv: float,
        rv: float,
    ) -> np.ndarray:
        """
        Fitzpatrick 1999 extinction law.
        Simplified version - for full implementation use external library.
        """
        # For simplicity, fall back to CCM89
        warnings.warn(
            "Using CCM89 as approximation for Fitzpatrick99",
            category=UserWarning,
        )
        return self._ccm89_extinction(wavelength, ebv, rv)

    def estimate_ebv_from_colors(
        self,
        observed_color: float,
        intrinsic_color: float,
        color_type: str = "B-V",
    ) -> float:
        """
        Estimate E(B-V) from observed and intrinsic colors.

        Parameters
        ----------
        observed_color : float
            Observed color
        intrinsic_color : float
            Intrinsic (unreddened) color
        color_type : str
            Type of color ('B-V', 'g-r', etc.)

        Returns
        -------
        float
            Estimated E(B-V)
        """
        if color_type == "B-V":
            # E(B-V) is directly the difference
            ebv = observed_color - intrinsic_color
        else:
            # For other colors, need conversion factors
            # This is a simplified approach
            ebv = (observed_color - intrinsic_color) / 1.0  # Placeholder
            warnings.warn(
                f"Color type {color_type} conversion is approximate",
                category=UserWarning,
            )

        return max(0.0, ebv)  # Ensure non-negative
