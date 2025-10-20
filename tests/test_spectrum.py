"""
Tests for the spectrum module.
"""

import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from lvmstars.spectrum import StellarSpectrum


def test_stellar_spectrum_creation():
    """Test basic StellarSpectrum creation."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = np.ones_like(wavelength)

    spectrum = StellarSpectrum(wavelength=wavelength, flux=flux)

    assert len(spectrum.wavelength) == 1000
    assert len(spectrum.flux) == 1000
    assert len(spectrum.flux_error) == 1000


def test_stellar_spectrum_with_coordinates():
    """Test StellarSpectrum with coordinates."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = np.ones_like(wavelength)
    coords = SkyCoord(ra=180.0 * u.deg, dec=30.0 * u.deg)

    spectrum = StellarSpectrum(wavelength=wavelength, flux=flux, coordinates=coords)

    assert spectrum.coordinates is not None
    assert spectrum.coordinates.ra.deg == 180.0
    assert spectrum.coordinates.dec.deg == 30.0


def test_from_lvm_spaxel():
    """Test creation from LVM spaxel data."""
    spaxel_data = {
        "wavelength": np.linspace(3600, 9800, 1000),
        "flux": np.ones(1000),
        "flux_error": 0.1 * np.ones(1000),
        "ra": 180.0,
        "dec": 30.0,
        "spaxel_id": 12345,
    }

    spectrum = StellarSpectrum.from_lvm_spaxel(spaxel_data)

    assert len(spectrum.wavelength) == 1000
    assert spectrum.coordinates.ra.deg == 180.0
    assert spectrum.metadata["spaxel_id"] == 12345


def test_set_and_get_parameters():
    """Test setting and getting stellar parameters."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = np.ones_like(wavelength)
    spectrum = StellarSpectrum(wavelength=wavelength, flux=flux)

    spectrum.set_parameters(teff=5800, logg=4.5, feh=0.0, alpha_fe=0.1)
    params = spectrum.get_parameters()

    assert params["teff"] == 5800
    assert params["logg"] == 4.5
    assert params["feh"] == 0.0
    assert params["alpha_fe"] == 0.1


def test_normalize_median():
    """Test median normalization."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = 2.0 * np.ones_like(wavelength)
    spectrum = StellarSpectrum(wavelength=wavelength, flux=flux)

    normalized = spectrum.normalize(method="median")

    assert np.median(normalized.flux) == pytest.approx(1.0)


def test_normalize_mean():
    """Test mean normalization."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = 3.0 * np.ones_like(wavelength)
    spectrum = StellarSpectrum(wavelength=wavelength, flux=flux)

    normalized = spectrum.normalize(method="mean")

    assert np.mean(normalized.flux) == pytest.approx(1.0)


def test_invalid_wavelength_flux_length():
    """Test that mismatched wavelength and flux raise error."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = np.ones(500)

    with pytest.raises(ValueError, match="same length"):
        StellarSpectrum(wavelength=wavelength, flux=flux)


def test_save_to_library(tmp_path):
    """Test saving spectrum to FITS file."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = np.ones_like(wavelength)
    spectrum = StellarSpectrum(wavelength=wavelength, flux=flux)
    spectrum.set_parameters(teff=5800, logg=4.5, feh=0.0)

    output_file = tmp_path / "test_spectrum.fits"
    spectrum.save_to_library(str(output_file))

    assert output_file.exists()
