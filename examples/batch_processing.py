"""
Example: Batch processing of multiple LVM spaxels.

This script demonstrates processing multiple spaxels at once,
including filtering by GAIA matches and organizing the output.
"""

import numpy as np

from lvmstars import (
    StellarSpectrum,
    SkySubtractor,
    DustCorrector,
    StellarParameters,
)
from lvmstars.parameters import ParameterGrid


def create_mock_spaxel_cube(n_spaxels=10):
    """Create mock spaxel cube data."""
    wavelength = np.linspace(3600, 9800, 2000)

    spaxels = []
    for i in range(n_spaxels):
        # Random position around field center
        ra = 180.0 + 0.01 * np.random.randn()
        dec = 30.0 + 0.01 * np.random.randn()

        # Mock spectrum
        flux = 1e-16 * np.exp(-(((wavelength - 5500) / 1000) ** 2)) + 1e-17
        flux *= 1 + 0.1 * np.random.randn()  # Add noise
        flux_error = 0.05 * np.abs(flux)

        spaxel_data = {
            "wavelength": wavelength,
            "flux": flux,
            "flux_error": flux_error,
            "ra": ra,
            "dec": dec,
            "spaxel_id": i,
        }

        spaxels.append(spaxel_data)

    return spaxels


def process_spaxel_batch(spaxel_data_list, output_dir="/tmp/stellar_library"):
    """Process a batch of spaxels through the full pipeline."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(spaxel_data_list)} spaxels...")
    print("=" * 60)

    # Initialize processing modules
    subtractor = SkySubtractor(method="median")
    corrector = DustCorrector(extinction_law="ccm89")

    # Convert to StellarSpectrum objects
    spectra = [StellarSpectrum.from_lvm_spaxel(data) for data in spaxel_data_list]

    # For this example, we'll use mock GAIA matches
    processed_count = 0
    single_star_count = 0

    for i, spectrum in enumerate(spectra):
        print(f"\nProcessing spaxel {i+1}/{len(spectra)}...")

        # Mock GAIA match (in real use, query GAIA)
        is_single = np.random.rand() > 0.3  # 70% single stars
        if not is_single:
            print("  Not a single star - skipping")
            continue

        single_star_count += 1

        # Mock nearby spaxels for sky subtraction
        nearby_indices = [j for j in range(len(spectra)) if abs(j - i) in [1, 2] and j != i]
        nearby = [spectra[j] for j in nearby_indices[:3]]

        if len(nearby) > 0:
            cleaned = subtractor.subtract_nearby_spaxels(spectrum, nearby)
        else:
            cleaned = spectrum

        # Mock dust correction
        dap_results = {"ebv": 0.05 * np.random.rand()}
        corrected = corrector.correct(cleaned, dap_results)

        # Mock stellar parameters
        teff = 4000 + 6000 * np.random.rand()
        logg = 2.0 + 3.0 * np.random.rand()
        feh = -1.0 + 1.5 * np.random.rand()
        alpha_fe = -0.2 + 0.4 * np.random.rand()

        corrected.set_parameters(teff=teff, logg=logg, feh=feh, alpha_fe=alpha_fe)

        # Save to library
        output_file = f"{output_dir}/spectrum_{i:04d}.fits"
        corrected.save_to_library(output_file)

        params = StellarParameters.from_dict(corrected.get_parameters())
        print(f"  Parameters: {params}")
        print(f"  Spectral type: {params.spectral_type_estimate()}")
        print(f"  Saved: {output_file}")

        processed_count += 1

    print("\n" + "=" * 60)
    print("Batch processing complete!")
    print(f"Total spaxels: {len(spectra)}")
    print(f"Single stars identified: {single_star_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Output directory: {output_dir}")

    return processed_count


def organize_by_parameters(spectra_files, grid):
    """
    Organize processed spectra into parameter grid bins.

    Parameters
    ----------
    spectra_files : list of str
        List of FITS files
    grid : ParameterGrid
        Parameter grid for organization
    """
    print("\n" + "=" * 60)
    print("Organizing spectra by parameters...")

    from astropy.io import fits

    grid_bins = {}

    for filename in spectra_files:
        try:
            with fits.open(filename) as hdul:
                header = hdul[0].header
                teff = header.get("TEFF")
                logg = header.get("LOGG")
                feh = header.get("FEH")

                if teff and logg and feh:
                    nearest = grid.find_nearest_grid_point(teff, logg, feh)
                    key = (nearest["teff"], nearest["logg"], nearest["feh"])

                    if key not in grid_bins:
                        grid_bins[key] = []
                    grid_bins[key].append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Organized into {len(grid_bins)} parameter bins:")
    for key, files in sorted(grid_bins.items())[:5]:
        print(f"  Teff={key[0]:.0f}K, logg={key[1]:.1f}, [Fe/H]={key[2]:.1f}: {len(files)} spectra")
    if len(grid_bins) > 5:
        print(f"  ... and {len(grid_bins) - 5} more bins")


def main():
    """Run batch processing example."""
    print("=" * 60)
    print("LVM Stellar Library - Batch Processing Example")
    print("=" * 60)
    print()

    # Create mock data
    print("Creating mock spaxel cube...")
    spaxel_data = create_mock_spaxel_cube(n_spaxels=20)
    print(f"Created {len(spaxel_data)} spaxels")
    print()

    # Process batch
    output_dir = "/tmp/stellar_library_batch"
    processed_count = process_spaxel_batch(spaxel_data, output_dir)

    # Organize by parameters
    if processed_count > 0:
        import glob

        spectra_files = glob.glob(f"{output_dir}/*.fits")

        if spectra_files:
            grid = ParameterGrid(
                teff_range=(4000, 10000),
                teff_step=500,
                logg_range=(2.0, 5.0),
                logg_step=0.5,
                feh_range=(-1.0, 0.5),
                feh_step=0.5,
            )
            organize_by_parameters(spectra_files, grid)


if __name__ == "__main__":
    main()
