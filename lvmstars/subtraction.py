"""
Sky and ISM residual subtraction from nearby spaxels.
"""

import numpy as np
from typing import List, Optional
from .spectrum import StellarSpectrum
import warnings


class SkySubtractor:
    """
    Subtract sky and ISM residuals from stellar spectra using nearby spaxels.
    """

    def __init__(self, method: str = "median", outlier_sigma: float = 3.0):
        """
        Initialize the SkySubtractor.

        Parameters
        ----------
        method : str
            Method for combining nearby spaxels ('median', 'mean', or 'weighted')
        outlier_sigma : float
            Sigma threshold for outlier rejection
        """
        self.method = method
        self.outlier_sigma = outlier_sigma

    def subtract_nearby_spaxels(
        self,
        target_spectrum: StellarSpectrum,
        nearby_spaxels: List[StellarSpectrum],
        weights: Optional[np.ndarray] = None,
    ) -> StellarSpectrum:
        """
        Subtract sky/ISM residuals estimated from nearby spaxels.

        Parameters
        ----------
        target_spectrum : StellarSpectrum
            The target stellar spectrum
        nearby_spaxels : list of StellarSpectrum
            Nearby spaxel spectra to estimate sky/ISM
        weights : np.ndarray, optional
            Weights for each nearby spaxel (for 'weighted' method)

        Returns
        -------
        StellarSpectrum
            Cleaned spectrum with sky/ISM subtracted
        """
        if len(nearby_spaxels) == 0:
            warnings.warn("No nearby spaxels provided, returning original spectrum")
            return target_spectrum

        # Check wavelength consistency
        target_wave = target_spectrum.wavelength
        for i, spaxel in enumerate(nearby_spaxels):
            if not np.allclose(spaxel.wavelength, target_wave, rtol=1e-6):
                raise ValueError(f"Nearby spaxel {i} has incompatible wavelength grid")

        # Stack nearby spaxel fluxes
        nearby_fluxes = np.array([spaxel.flux for spaxel in nearby_spaxels])

        # Estimate sky/ISM residual
        if self.method == "median":
            sky_residual = np.median(nearby_fluxes, axis=0)
        elif self.method == "mean":
            # Use sigma-clipped mean for outlier rejection
            sky_residual = self._sigma_clipped_mean(nearby_fluxes)
        elif self.method == "weighted":
            if weights is None:
                raise ValueError("Weights must be provided for 'weighted' method")
            if len(weights) != len(nearby_spaxels):
                raise ValueError("Number of weights must match number of nearby spaxels")
            weights = np.asarray(weights)
            weights = weights / np.sum(weights)  # Normalize
            sky_residual = np.average(nearby_fluxes, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Subtract sky residual from target
        cleaned_flux = target_spectrum.flux - sky_residual

        # Propagate uncertainties
        if self.method == "median":
            # Estimate uncertainty from MAD
            mad = np.median(np.abs(nearby_fluxes - sky_residual), axis=0)
            sky_error = 1.4826 * mad  # Convert MAD to std
        elif self.method == "mean":
            sky_error = np.std(nearby_fluxes, axis=0) / np.sqrt(len(nearby_spaxels))
        elif self.method == "weighted":
            # Weighted variance
            variance = np.average((nearby_fluxes - sky_residual) ** 2, axis=0, weights=weights)
            sky_error = np.sqrt(variance)

        # Combine uncertainties
        cleaned_error = np.sqrt(target_spectrum.flux_error**2 + sky_error**2)

        # Create cleaned spectrum
        cleaned_spectrum = StellarSpectrum(
            wavelength=target_spectrum.wavelength.copy(),
            flux=cleaned_flux,
            flux_error=cleaned_error,
            coordinates=target_spectrum.coordinates,
            metadata=target_spectrum.metadata.copy(),
        )

        # Copy stellar parameters if they exist
        cleaned_spectrum.parameters = target_spectrum.parameters.copy()

        # Add subtraction metadata
        cleaned_spectrum.metadata["sky_subtraction"] = {
            "method": self.method,
            "n_nearby_spaxels": len(nearby_spaxels),
        }

        return cleaned_spectrum

    def _sigma_clipped_mean(self, fluxes: np.ndarray) -> np.ndarray:
        """
        Compute sigma-clipped mean along axis 0.

        Parameters
        ----------
        fluxes : np.ndarray
            Array of fluxes with shape (n_spaxels, n_wavelength)

        Returns
        -------
        np.ndarray
            Sigma-clipped mean
        """
        mean = np.mean(fluxes, axis=0)
        std = np.std(fluxes, axis=0)

        # Create mask for outliers
        mask = np.abs(fluxes - mean) < self.outlier_sigma * std

        # Compute masked mean
        masked_fluxes = np.where(mask, fluxes, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            clipped_mean = np.nanmean(masked_fluxes, axis=0)

        # Fill any NaN values with original mean
        clipped_mean = np.where(np.isnan(clipped_mean), mean, clipped_mean)

        return clipped_mean

    def estimate_sky_from_mask(
        self,
        spectra: List[StellarSpectrum],
        stellar_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate sky/ISM residual from spaxels marked as non-stellar.

        Parameters
        ----------
        spectra : list of StellarSpectrum
            All spaxel spectra
        stellar_mask : np.ndarray
            Boolean mask where True indicates stellar spaxel

        Returns
        -------
        np.ndarray
            Estimated sky/ISM spectrum
        """
        if len(spectra) != len(stellar_mask):
            raise ValueError("Length of spectra and mask must match")

        # Select non-stellar spaxels
        sky_spaxels = [spec for i, spec in enumerate(spectra) if not stellar_mask[i]]

        if len(sky_spaxels) == 0:
            warnings.warn("No sky spaxels found in mask")
            return np.zeros_like(spectra[0].wavelength)

        # Stack and combine
        sky_fluxes = np.array([spaxel.flux for spaxel in sky_spaxels])

        if self.method == "median":
            sky_spectrum = np.median(sky_fluxes, axis=0)
        elif self.method == "mean":
            sky_spectrum = self._sigma_clipped_mean(sky_fluxes)
        else:
            sky_spectrum = np.mean(sky_fluxes, axis=0)

        return sky_spectrum

    def subtract_background(
        self,
        spectrum: StellarSpectrum,
        background_level: float,
    ) -> StellarSpectrum:
        """
        Subtract a constant background level from spectrum.

        Parameters
        ----------
        spectrum : StellarSpectrum
            Input spectrum
        background_level : float
            Background level to subtract

        Returns
        -------
        StellarSpectrum
            Background-subtracted spectrum
        """
        cleaned_flux = spectrum.flux - background_level

        return StellarSpectrum(
            wavelength=spectrum.wavelength.copy(),
            flux=cleaned_flux,
            flux_error=spectrum.flux_error.copy(),
            coordinates=spectrum.coordinates,
            metadata=spectrum.metadata.copy(),
        )
