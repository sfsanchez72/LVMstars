"""
Core spectral data structures for LVM stellar library.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Optional, Dict, Any
import warnings


class StellarSpectrum:
    """
    Represents a stellar spectrum from an LVM spaxel.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array in Angstroms
    flux : np.ndarray
        Flux array
    flux_error : np.ndarray
        Flux uncertainty array
    coordinates : SkyCoord
        Sky coordinates of the spaxel
    metadata : dict
        Additional metadata about the spectrum
    parameters : dict
        Stellar parameters (Teff, log g, [Fe/H], [α/Fe])
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        flux_error: Optional[np.ndarray] = None,
        coordinates: Optional[SkyCoord] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a StellarSpectrum object.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in Angstroms
        flux : np.ndarray
            Flux array
        flux_error : np.ndarray, optional
            Flux uncertainty array
        coordinates : SkyCoord, optional
            Sky coordinates of the spaxel
        metadata : dict, optional
            Additional metadata
        """
        self.wavelength = np.asarray(wavelength)
        self.flux = np.asarray(flux)

        if flux_error is not None:
            self.flux_error = np.asarray(flux_error)
        else:
            self.flux_error = np.zeros_like(self.flux)

        self.coordinates = coordinates
        self.metadata = metadata or {}
        self.parameters = {}

        # Validate shapes
        if len(self.wavelength) != len(self.flux):
            raise ValueError("Wavelength and flux arrays must have same length")
        if len(self.flux_error) != len(self.flux):
            raise ValueError("Flux error array must have same length as flux array")

    @classmethod
    def from_lvm_spaxel(cls, spaxel_data: Dict[str, Any]) -> "StellarSpectrum":
        """
        Create a StellarSpectrum from LVM spaxel data.

        Parameters
        ----------
        spaxel_data : dict
            Dictionary containing spaxel data with keys:
            - 'wavelength': wavelength array
            - 'flux': flux array
            - 'flux_error': flux error array (optional)
            - 'ra': right ascension in degrees
            - 'dec': declination in degrees
            - other metadata

        Returns
        -------
        StellarSpectrum
            New spectrum object
        """
        wavelength = spaxel_data.get("wavelength")
        flux = spaxel_data.get("flux")
        flux_error = spaxel_data.get("flux_error")

        # Create coordinates if RA/Dec are provided
        coordinates = None
        if "ra" in spaxel_data and "dec" in spaxel_data:
            coordinates = SkyCoord(
                ra=spaxel_data["ra"] * u.deg,
                dec=spaxel_data["dec"] * u.deg,
                frame="icrs",
            )

        # Collect other metadata
        metadata = {
            k: v
            for k, v in spaxel_data.items()
            if k not in ["wavelength", "flux", "flux_error", "ra", "dec"]
        }

        return cls(
            wavelength=wavelength,
            flux=flux,
            flux_error=flux_error,
            coordinates=coordinates,
            metadata=metadata,
        )

    def set_parameters(
        self,
        teff: Optional[float] = None,
        logg: Optional[float] = None,
        feh: Optional[float] = None,
        alpha_fe: Optional[float] = None,
    ):
        """
        Set stellar parameters for this spectrum.

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
        """
        if teff is not None:
            self.parameters["teff"] = teff
        if logg is not None:
            self.parameters["logg"] = logg
        if feh is not None:
            self.parameters["feh"] = feh
        if alpha_fe is not None:
            self.parameters["alpha_fe"] = alpha_fe

    def get_parameters(self) -> Dict[str, float]:
        """
        Get stellar parameters.

        Returns
        -------
        dict
            Dictionary of stellar parameters
        """
        return self.parameters.copy()

    def normalize(self, method: str = "median") -> "StellarSpectrum":
        """
        Normalize the spectrum.

        Parameters
        ----------
        method : str
            Normalization method ('median', 'mean', or 'continuum')

        Returns
        -------
        StellarSpectrum
            Normalized spectrum (new object)
        """
        if method == "median":
            norm_factor = np.median(self.flux)
        elif method == "mean":
            norm_factor = np.mean(self.flux)
        elif method == "continuum":
            # Simple continuum estimation (could be improved)
            from scipy.signal import medfilt

            continuum = medfilt(self.flux, kernel_size=51)
            norm_factor = continuum
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if np.isscalar(norm_factor):
            if norm_factor == 0:
                warnings.warn("Normalization factor is zero, skipping normalization")
                return self
            normalized_flux = self.flux / norm_factor
            normalized_error = self.flux_error / norm_factor
        else:
            # Array normalization (continuum case)
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_flux = np.where(norm_factor != 0, self.flux / norm_factor, 0)
                normalized_error = np.where(norm_factor != 0, self.flux_error / norm_factor, 0)

        return StellarSpectrum(
            wavelength=self.wavelength.copy(),
            flux=normalized_flux,
            flux_error=normalized_error,
            coordinates=self.coordinates,
            metadata=self.metadata.copy(),
        )

    def save_to_library(self, filename: str):
        """
        Save spectrum to the stellar library.

        Parameters
        ----------
        filename : str
            Output filename
        """
        from astropy.io import fits

        # Create FITS file with spectrum and metadata
        primary_hdu = fits.PrimaryHDU()

        # Add parameters to header
        for key, value in self.parameters.items():
            primary_hdu.header[key.upper()] = value

        # Add coordinates
        if self.coordinates is not None:
            primary_hdu.header["RA"] = self.coordinates.ra.deg
            primary_hdu.header["DEC"] = self.coordinates.dec.deg

        # Add metadata
        for key, value in self.metadata.items():
            try:
                primary_hdu.header[key.upper()[:8]] = value
            except (ValueError, TypeError):
                # Skip values that can't be serialized to FITS
                pass

        # Create table with spectrum data
        col1 = fits.Column(name="wavelength", format="D", array=self.wavelength)
        col2 = fits.Column(name="flux", format="D", array=self.flux)
        col3 = fits.Column(name="flux_error", format="D", array=self.flux_error)

        table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])

        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(filename, overwrite=True)

    def __repr__(self):
        coord_str = ""
        if self.coordinates is not None:
            coord_str = f" at {self.coordinates.to_string('hmsdms')}"
        return f"StellarSpectrum(λ={self.wavelength[0]:.1f}-{self.wavelength[-1]:.1f} Å{coord_str})"
