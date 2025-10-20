"""
Example workflow for building LVM stellar library.

This script demonstrates the complete pipeline:
1. Load LVM spaxel data
2. Cross-match with GAIA
3. Subtract sky/ISM residuals
4. Apply dust correction
5. Tag with stellar parameters
6. Save to library
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from lvmstars import (
    StellarSpectrum,
    GaiaCrossmatcher,
    SkySubtractor,
    DustCorrector,
    StellarParameters,
)


def create_mock_lvm_data():
    """Create mock LVM spaxel data for demonstration."""
    # Create mock wavelength array (typical LVM range)
    wavelength = np.linspace(3600, 9800, 2000)
    
    # Create mock flux (simple blackbody-like spectrum)
    flux = 1e-16 * np.exp(-((wavelength - 5500) / 1000) ** 2) + 1e-17
    flux_error = 0.05 * flux
    
    # Mock spaxel data
    spaxel_data = {
        "wavelength": wavelength,
        "flux": flux,
        "flux_error": flux_error,
        "ra": 180.0,  # degrees
        "dec": 30.0,  # degrees
        "spaxel_id": 12345,
    }
    
    return spaxel_data


def create_mock_nearby_spaxels(n_spaxels=5):
    """Create mock nearby spaxel spectra for sky subtraction."""
    wavelength = np.linspace(3600, 9800, 2000)
    
    nearby = []
    for i in range(n_spaxels):
        # Create low-level sky/ISM residual
        flux = 1e-18 * (1 + 0.1 * np.random.randn(len(wavelength)))
        flux_error = 0.1 * np.abs(flux)
        
        spectrum = StellarSpectrum(
            wavelength=wavelength,
            flux=flux,
            flux_error=flux_error,
            coordinates=SkyCoord(
                ra=180.0 + 0.001 * (i + 1), dec=30.0 + 0.001 * (i + 1), unit="deg"
            ),
        )
        nearby.append(spectrum)
    
    return nearby


def main():
    """Run the example workflow."""
    print("=" * 60)
    print("LVM Stellar Library - Example Workflow")
    print("=" * 60)
    print()
    
    # Step 1: Load LVM spaxel data
    print("Step 1: Loading LVM spaxel data...")
    spaxel_data = create_mock_lvm_data()
    spectrum = StellarSpectrum.from_lvm_spaxel(spaxel_data)
    print(f"  Loaded: {spectrum}")
    print(f"  Wavelength range: {spectrum.wavelength[0]:.1f} - {spectrum.wavelength[-1]:.1f} Å")
    print()
    
    # Step 2: Cross-match with GAIA
    print("Step 2: Cross-matching with GAIA catalog...")
    print("  (Note: This requires internet connection and may take a moment)")
    print("  Skipping GAIA query in this example - using mock data")
    
    # Create mock GAIA match
    class MockGaiaMatch:
        def __init__(self):
            self.source_id = 987654321
            self.separation = 0.5  # arcsec
            self.n_neighbors = 0
            self.teff = 5800
            self.logg = 4.5
            self.feh = 0.0
            self.alpha_fe = 0.0
        
        def is_single_star(self, **kwargs):
            return True
        
        def has_parameters(self):
            return True
    
    gaia_match = MockGaiaMatch()
    print(f"  GAIA source ID: {gaia_match.source_id}")
    print(f"  Separation: {gaia_match.separation:.2f} arcsec")
    print(f"  Single star: {gaia_match.is_single_star()}")
    print()
    
    # Only proceed if single star
    if gaia_match.is_single_star():
        # Step 3: Subtract sky/ISM residuals
        print("Step 3: Subtracting sky/ISM residuals...")
        nearby_spaxels = create_mock_nearby_spaxels(n_spaxels=5)
        subtractor = SkySubtractor(method="median")
        cleaned_spectrum = subtractor.subtract_nearby_spaxels(
            spectrum, nearby_spaxels
        )
        print(f"  Used {len(nearby_spaxels)} nearby spaxels")
        print(f"  Method: {subtractor.method}")
        print()
        
        # Step 4: Apply dust correction
        print("Step 4: Applying dust correction...")
        corrector = DustCorrector(extinction_law="ccm89")
        dap_results = {"ebv": 0.05}  # Mock E(B-V) from DAP
        corrected_spectrum = corrector.correct(
            cleaned_spectrum, dap_results, rv=3.1
        )
        print(f"  E(B-V): {dap_results['ebv']:.3f}")
        print(f"  Extinction law: {corrector.extinction_law}")
        print()
        
        # Step 5: Tag with stellar parameters
        print("Step 5: Tagging with stellar parameters...")
        corrected_spectrum.set_parameters(
            teff=gaia_match.teff,
            logg=gaia_match.logg,
            feh=gaia_match.feh,
            alpha_fe=gaia_match.alpha_fe,
        )
        params = StellarParameters.from_dict(corrected_spectrum.get_parameters())
        print(f"  {params}")
        print(f"  Spectral type estimate: {params.spectral_type_estimate()}")
        print(f"  Parameters complete: {params.is_complete()}")
        print(f"  Parameters valid: {params.validate()}")
        print()
        
        # Step 6: Save to library
        print("Step 6: Saving to library...")
        output_file = "/tmp/stellar_spectrum_example.fits"
        corrected_spectrum.save_to_library(output_file)
        print(f"  Saved to: {output_file}")
        print()
        
        # Summary
        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  - Processed spectrum: {corrected_spectrum}")
        print(f"  - GAIA source: {gaia_match.source_id}")
        print(f"  - Stellar parameters: {params}")
        print(f"  - Output file: {output_file}")
        
    else:
        print("  Not a single star - skipping further processing")


if __name__ == "__main__":
    main()
