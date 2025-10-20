"""
Tests for the subtraction module.
"""

import pytest
import numpy as np
from lvmstars.spectrum import StellarSpectrum
from lvmstars.subtraction import SkySubtractor


def create_test_spectrum(flux_level=1.0):
    """Helper to create a test spectrum."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = flux_level * np.ones_like(wavelength)
    flux_error = 0.1 * flux
    return StellarSpectrum(wavelength=wavelength, flux=flux, flux_error=flux_error)


def test_sky_subtractor_creation():
    """Test SkySubtractor creation."""
    subtractor = SkySubtractor(method="median")
    assert subtractor.method == "median"
    assert subtractor.outlier_sigma == 3.0


def test_subtract_nearby_spaxels_median():
    """Test sky subtraction with median method."""
    target = create_test_spectrum(flux_level=10.0)
    nearby = [create_test_spectrum(flux_level=1.0) for _ in range(5)]
    
    subtractor = SkySubtractor(method="median")
    cleaned = subtractor.subtract_nearby_spaxels(target, nearby)
    
    # Should subtract ~1.0 from 10.0, leaving ~9.0
    assert np.median(cleaned.flux) == pytest.approx(9.0, rel=0.01)


def test_subtract_nearby_spaxels_mean():
    """Test sky subtraction with mean method."""
    target = create_test_spectrum(flux_level=10.0)
    nearby = [create_test_spectrum(flux_level=2.0) for _ in range(5)]
    
    subtractor = SkySubtractor(method="mean")
    cleaned = subtractor.subtract_nearby_spaxels(target, nearby)
    
    # Should subtract ~2.0 from 10.0, leaving ~8.0
    assert np.mean(cleaned.flux) == pytest.approx(8.0, rel=0.01)


def test_subtract_nearby_spaxels_weighted():
    """Test sky subtraction with weighted method."""
    target = create_test_spectrum(flux_level=10.0)
    nearby = [
        create_test_spectrum(flux_level=1.0),
        create_test_spectrum(flux_level=3.0),
    ]
    weights = np.array([0.8, 0.2])
    
    subtractor = SkySubtractor(method="weighted")
    cleaned = subtractor.subtract_nearby_spaxels(target, nearby, weights=weights)
    
    # Weighted average: 0.8*1.0 + 0.2*3.0 = 1.4
    # Result: 10.0 - 1.4 = 8.6
    assert np.mean(cleaned.flux) == pytest.approx(8.6, rel=0.01)


def test_subtract_no_nearby_spaxels():
    """Test behavior with no nearby spaxels."""
    target = create_test_spectrum(flux_level=10.0)
    
    subtractor = SkySubtractor(method="median")
    with pytest.warns(UserWarning, match="No nearby spaxels"):
        cleaned = subtractor.subtract_nearby_spaxels(target, [])
    
    # Should return original spectrum
    assert np.array_equal(cleaned.flux, target.flux)


def test_subtract_incompatible_wavelength():
    """Test that incompatible wavelength grids raise error."""
    target = create_test_spectrum(flux_level=10.0)
    
    # Create nearby spaxel with different wavelength grid
    wavelength = np.linspace(4000, 9000, 500)
    flux = np.ones_like(wavelength)
    nearby_bad = StellarSpectrum(wavelength=wavelength, flux=flux)
    
    subtractor = SkySubtractor(method="median")
    with pytest.raises((ValueError, AssertionError)):
        subtractor.subtract_nearby_spaxels(target, [nearby_bad])


def test_subtract_background():
    """Test constant background subtraction."""
    spectrum = create_test_spectrum(flux_level=10.0)
    
    subtractor = SkySubtractor()
    cleaned = subtractor.subtract_background(spectrum, background_level=2.0)
    
    assert np.mean(cleaned.flux) == pytest.approx(8.0)


def test_metadata_added():
    """Test that subtraction metadata is added."""
    target = create_test_spectrum(flux_level=10.0)
    nearby = [create_test_spectrum(flux_level=1.0) for _ in range(3)]
    
    subtractor = SkySubtractor(method="median")
    cleaned = subtractor.subtract_nearby_spaxels(target, nearby)
    
    assert "sky_subtraction" in cleaned.metadata
    assert cleaned.metadata["sky_subtraction"]["method"] == "median"
    assert cleaned.metadata["sky_subtraction"]["n_nearby_spaxels"] == 3
