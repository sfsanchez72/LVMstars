"""
Tests for the dust correction module.
"""

import pytest
import numpy as np
from lvmstars.spectrum import StellarSpectrum
from lvmstars.dust import DustCorrector


def create_test_spectrum():
    """Helper to create a test spectrum."""
    wavelength = np.linspace(3600, 9800, 1000)
    flux = np.ones_like(wavelength)
    flux_error = 0.1 * flux
    return StellarSpectrum(wavelength=wavelength, flux=flux, flux_error=flux_error)


def test_dust_corrector_creation():
    """Test DustCorrector creation."""
    corrector = DustCorrector(extinction_law="ccm89")
    assert corrector.extinction_law == "ccm89"


def test_dust_correction_with_ebv():
    """Test dust correction with E(B-V)."""
    spectrum = create_test_spectrum()
    corrector = DustCorrector(extinction_law="ccm89")

    dap_results = {"ebv": 0.1}
    corrected = corrector.correct(spectrum, dap_results, rv=3.1)

    # Corrected flux should be higher than original (dereddening)
    assert np.mean(corrected.flux) > np.mean(spectrum.flux)


def test_dust_correction_zero_ebv():
    """Test that zero E(B-V) returns unchanged spectrum."""
    spectrum = create_test_spectrum()
    corrector = DustCorrector(extinction_law="ccm89")

    dap_results = {"ebv": 0.0}
    corrected = corrector.correct(spectrum, dap_results)

    # Should return original spectrum
    assert np.array_equal(corrected.flux, spectrum.flux)


def test_dust_correction_no_ebv():
    """Test behavior when no E(B-V) is provided."""
    spectrum = create_test_spectrum()
    corrector = DustCorrector(extinction_law="ccm89")

    dap_results = {}
    with pytest.warns(UserWarning, match="No E\\(B-V\\)"):
        corrected = corrector.correct(spectrum, dap_results)

    # Should return original spectrum
    assert np.array_equal(corrected.flux, spectrum.flux)


def test_dust_correction_override_ebv():
    """Test that explicit ebv parameter overrides dap_results."""
    spectrum = create_test_spectrum()
    corrector = DustCorrector(extinction_law="ccm89")

    dap_results = {"ebv": 0.05}
    corrected = corrector.correct(spectrum, dap_results, ebv=0.1)

    # Check metadata uses the override value
    assert corrected.metadata["dust_correction"]["ebv"] == 0.1


def test_ccm89_extinction():
    """Test CCM89 extinction calculation."""
    corrector = DustCorrector(extinction_law="ccm89")

    wavelength = np.array([4400.0, 5500.0, 6500.0])  # B, V, R bands
    ebv = 0.1
    rv = 3.1

    extinction = corrector._ccm89_extinction(wavelength, ebv, rv)

    # Check that extinction decreases with wavelength
    assert extinction[0] > extinction[1] > extinction[2]
    # Check reasonable values
    assert 0.0 < extinction[1] < 1.0  # A_V for E(B-V)=0.1


def test_calzetti_extinction():
    """Test Calzetti extinction calculation."""
    corrector = DustCorrector(extinction_law="calzetti00")

    wavelength = np.array([4400.0, 5500.0, 6500.0])
    ebv = 0.1

    extinction = corrector._calzetti00_extinction(wavelength, ebv, rv=4.05)

    # Check that extinction decreases with wavelength
    assert extinction[0] > extinction[1] > extinction[2]


def test_metadata_added():
    """Test that correction metadata is added."""
    spectrum = create_test_spectrum()
    corrector = DustCorrector(extinction_law="ccm89")

    dap_results = {"ebv": 0.1}
    corrected = corrector.correct(spectrum, dap_results, rv=3.1)

    assert "dust_correction" in corrected.metadata
    assert corrected.metadata["dust_correction"]["ebv"] == 0.1
    assert corrected.metadata["dust_correction"]["rv"] == 3.1
    assert corrected.metadata["dust_correction"]["extinction_law"] == "ccm89"


def test_estimate_ebv_from_colors():
    """Test E(B-V) estimation from colors."""
    corrector = DustCorrector()

    # Reddened star: observed B-V = 0.8, intrinsic B-V = 0.6
    ebv = corrector.estimate_ebv_from_colors(0.8, 0.6, color_type="B-V")

    assert ebv == pytest.approx(0.2)


def test_estimate_ebv_negative():
    """Test that negative E(B-V) is clipped to zero."""
    corrector = DustCorrector()

    # Unreddened or mis-estimated: would give negative E(B-V)
    ebv = corrector.estimate_ebv_from_colors(0.5, 0.7, color_type="B-V")

    assert ebv == 0.0


def test_unknown_extinction_law():
    """Test that unknown extinction law raises error."""
    corrector = DustCorrector(extinction_law="unknown")
    spectrum = create_test_spectrum()

    dap_results = {"ebv": 0.1}
    with pytest.raises(ValueError, match="Unknown extinction law"):
        corrector.correct(spectrum, dap_results)
