"""
Tests for utility functions.
"""

import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from lvmstars.utils import (
    calculate_distances,
    find_nearby_spaxels,
    wavelength_to_velocity,
    velocity_to_wavelength,
    resample_spectrum,
    mask_bad_pixels,
    estimate_continuum,
    calculate_snr,
    create_wavelength_grid,
)


def test_calculate_distances():
    """Test angular distance calculation."""
    reference = SkyCoord(ra=180.0 * u.deg, dec=30.0 * u.deg)
    coord_list = [
        SkyCoord(ra=180.0 * u.deg, dec=30.0 * u.deg),
        SkyCoord(ra=180.0 * u.deg, dec=30.01 * u.deg),  # ~36 arcsec
    ]
    
    distances = calculate_distances(coord_list, reference)
    
    assert distances[0] == pytest.approx(0.0, abs=1e-6)
    assert distances[1] == pytest.approx(36.0, rel=0.1)


def test_find_nearby_spaxels():
    """Test finding nearby spaxels."""
    target = SkyCoord(ra=180.0 * u.deg, dec=30.0 * u.deg)
    spaxels = [
        SkyCoord(ra=180.0 * u.deg, dec=30.0 * u.deg),  # Same position
        SkyCoord(ra=180.0 * u.deg, dec=30.001 * u.deg),  # ~3.6 arcsec
        SkyCoord(ra=180.0 * u.deg, dec=30.01 * u.deg),  # ~36 arcsec
    ]
    
    nearby = find_nearby_spaxels(target, spaxels, radius=10.0, exclude_target=True)
    
    # Should find only index 1 (3.6 arcsec away)
    assert len(nearby) == 1
    assert nearby[0] == 1


def test_wavelength_to_velocity():
    """Test wavelength to velocity conversion."""
    rest_wave = 5000.0  # Angstroms
    wavelength = np.array([5000.0, 5005.0, 4995.0])
    
    velocity = wavelength_to_velocity(wavelength, rest_wave)
    
    assert velocity[0] == pytest.approx(0.0)
    assert velocity[1] > 0  # Redshifted
    assert velocity[2] < 0  # Blueshifted


def test_velocity_to_wavelength():
    """Test velocity to wavelength conversion."""
    rest_wave = 5000.0
    velocity = np.array([0.0, 100.0, -100.0])  # km/s
    
    wavelength = velocity_to_wavelength(velocity, rest_wave)
    
    assert wavelength[0] == pytest.approx(rest_wave)
    assert wavelength[1] > rest_wave
    assert wavelength[2] < rest_wave


def test_resample_spectrum():
    """Test spectrum resampling."""
    wavelength = np.linspace(4000, 6000, 100)
    flux = np.sin(wavelength / 100)
    new_wavelength = np.linspace(4500, 5500, 50)
    
    resampled_flux, _ = resample_spectrum(wavelength, flux, new_wavelength)
    
    assert len(resampled_flux) == 50
    assert np.all(np.isfinite(resampled_flux))


def test_mask_bad_pixels():
    """Test bad pixel masking."""
    flux = np.array([1.0, 2.0, np.inf, 3.0, np.nan])
    flux_error = np.array([0.1, 0.2, 0.1, 0.3, 0.1])
    
    mask = mask_bad_pixels(flux, flux_error, threshold=5.0)
    
    assert mask[0] == True  # Good pixel (SNR=10)
    assert mask[1] == True  # Good pixel (SNR=10)
    assert mask[2] == False  # Bad pixel (inf)
    assert mask[3] == True  # Good pixel (SNR=10)
    assert mask[4] == False  # Bad pixel (nan)


def test_estimate_continuum():
    """Test continuum estimation."""
    wavelength = np.linspace(4000, 6000, 1000)
    continuum_level = 2.0 + 0.0001 * wavelength
    absorption_line = np.exp(-((wavelength - 5000) / 10) ** 2)
    flux = continuum_level * (1 - 0.5 * absorption_line)
    
    estimated_continuum = estimate_continuum(wavelength, flux, window_size=51)
    
    # Check that estimated continuum is close to true continuum
    assert np.mean(np.abs(estimated_continuum - continuum_level)) < 0.5


def test_calculate_snr():
    """Test SNR calculation."""
    flux = 10.0 * np.ones(100)
    flux_error = 1.0 * np.ones(100)
    
    snr = calculate_snr(flux, flux_error)
    
    assert snr == pytest.approx(10.0)


def test_calculate_snr_with_wavelength_range():
    """Test SNR calculation with wavelength range."""
    wavelength = np.linspace(4000, 6000, 100)
    flux = 10.0 * np.ones(100)
    flux_error = 1.0 * np.ones(100)
    
    snr = calculate_snr(
        flux, flux_error, wavelength_range=(4500, 5500), wavelength=wavelength
    )
    
    assert snr == pytest.approx(10.0)


def test_create_wavelength_grid_linear():
    """Test linear wavelength grid creation."""
    grid = create_wavelength_grid(4000, 6000, n_pixels=100, log_scale=False)
    
    assert len(grid) == 100
    assert grid[0] == pytest.approx(4000)
    assert grid[-1] == pytest.approx(6000)


def test_create_wavelength_grid_log():
    """Test logarithmic wavelength grid creation."""
    grid = create_wavelength_grid(4000, 6000, n_pixels=100, log_scale=True)
    
    assert len(grid) == 100
    assert grid[0] == pytest.approx(4000)
    assert grid[-1] == pytest.approx(6000)


def test_create_wavelength_grid_with_resolution():
    """Test wavelength grid creation with resolution."""
    grid = create_wavelength_grid(4000, 6000, resolution=1000, log_scale=False)
    
    assert len(grid) > 0
    assert grid[0] == pytest.approx(4000, abs=10)
    assert grid[-1] == pytest.approx(6000, abs=10)


def test_create_wavelength_grid_no_params():
    """Test that grid creation without parameters raises error."""
    with pytest.raises(ValueError, match="resolution or n_pixels"):
        create_wavelength_grid(4000, 6000)
