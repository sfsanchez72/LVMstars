# LVM Stellar Library Pipeline

This document describes the complete pipeline for building the LVM stellar library.

## Pipeline Overview

The LVM stellar library pipeline consists of the following steps:

1. **Load LVM Spaxel Data**: Read spectral data from LVM observations
2. **GAIA Cross-matching**: Identify single-star spectra using GAIA catalog
3. **Sky/ISM Residual Subtraction**: Remove contamination from nearby spaxels
4. **Dust Correction**: Apply extinction correction using DAP results
5. **Parameter Tagging**: Associate spectra with stellar parameters (Teff, log g, [Fe/H], [α/Fe])
6. **Library Storage**: Save processed spectra to the stellar library

## Detailed Workflow

### 1. Load LVM Spaxel Data

```python
from lvmstars import StellarSpectrum

# From dictionary
spaxel_data = {
    "wavelength": wavelength_array,  # Angstroms
    "flux": flux_array,
    "flux_error": error_array,
    "ra": ra_deg,
    "dec": dec_deg,
}
spectrum = StellarSpectrum.from_lvm_spaxel(spaxel_data)
```

### 2. GAIA Cross-matching

```python
from lvmstars import GaiaCrossmatcher

matcher = GaiaCrossmatcher(gaia_release="gaiadr3")
gaia_match = matcher.match_spaxel(
    spectrum.coordinates,
    radius=1.0  # arcseconds
)

# Check if single star
if gaia_match and gaia_match.is_single_star():
    print("Single star identified")
    print(f"GAIA source_id: {gaia_match.source_id}")
```

### 3. Sky/ISM Residual Subtraction

```python
from lvmstars import SkySubtractor

subtractor = SkySubtractor(method="median")

# Get nearby spaxels (typically within a few arcseconds)
nearby_spaxels = [...]  # List of StellarSpectrum objects

cleaned_spectrum = subtractor.subtract_nearby_spaxels(
    spectrum,
    nearby_spaxels
)
```

### 4. Dust Correction

```python
from lvmstars import DustCorrector

corrector = DustCorrector(extinction_law="ccm89")

# DAP results containing E(B-V)
dap_results = {"ebv": 0.05}

corrected_spectrum = corrector.correct(
    cleaned_spectrum,
    dap_results,
    rv=3.1
)
```

### 5. Parameter Tagging

```python
from lvmstars import StellarParameters

# From GAIA match
corrected_spectrum.set_parameters(
    teff=gaia_match.teff,
    logg=gaia_match.logg,
    feh=gaia_match.feh,
    alpha_fe=gaia_match.alpha_fe
)

# Validate parameters
params = StellarParameters.from_dict(corrected_spectrum.get_parameters())
if params.validate() and params.is_complete():
    print(f"Valid parameters: {params}")
```

### 6. Library Storage

```python
# Save to FITS file
output_file = "stellar_library/spectrum_12345.fits"
corrected_spectrum.save_to_library(output_file)
```

## Quality Control

### Selection Criteria

- **GAIA Match**: Must be identified as a single star
  - `separation < 1.0` arcsec
  - `n_neighbors == 0` within search radius
  - Valid parallax measurement

- **Signal-to-Noise**: Minimum SNR threshold
  ```python
  from lvmstars.utils import calculate_snr
  snr = calculate_snr(spectrum.flux, spectrum.flux_error)
  if snr >= 5.0:
      # Process spectrum
  ```

- **Stellar Parameters**: Must have complete parameter set
  - Teff: 2000-50000 K
  - log g: -1 to 6
  - [Fe/H]: -5 to 1
  - [α/Fe]: -0.5 to 1 (optional)

## Configuration

The pipeline can be configured using a JSON configuration file:

```python
from lvmstars import Config

config = Config("config.json")

# Access configuration
gaia_release = config.get("gaia", "release")
extinction_law = config.get("dust_correction", "extinction_law")
```

Default configuration:
```json
{
  "gaia": {
    "release": "gaiadr3",
    "search_radius": 1.0,
    "max_neighbors": 0
  },
  "sky_subtraction": {
    "method": "median",
    "outlier_sigma": 3.0,
    "n_nearby_spaxels": 5
  },
  "dust_correction": {
    "extinction_law": "ccm89",
    "rv": 3.1
  },
  "quality": {
    "min_snr": 5.0,
    "max_separation": 1.0
  }
}
```

## Batch Processing

For processing multiple spaxels, use the batch processing approach:

```python
from lvmstars import StellarSpectrum, SkySubtractor, DustCorrector

# Load all spaxels
spectra = [StellarSpectrum.from_lvm_spaxel(data) for data in spaxel_data_list]

# Process each spaxel
for i, spectrum in enumerate(spectra):
    # Apply pipeline steps...
    processed_spectrum.save_to_library(f"library/spectrum_{i:04d}.fits")
```

See `examples/batch_processing.py` for a complete example.

## Organizing the Library

### Parameter Grid

Organize spectra by stellar parameters:

```python
from lvmstars.parameters import ParameterGrid

grid = ParameterGrid(
    teff_range=(4000, 10000),
    teff_step=250,
    logg_range=(2.0, 5.0),
    logg_step=0.5,
    feh_range=(-2.0, 0.5),
    feh_step=0.5
)

# Find nearest grid point
nearest = grid.find_nearest_grid_point(teff=5823, logg=4.47, feh=-0.23)
```

### Directory Structure

Recommended directory structure for the stellar library:

```
stellar_library/
├── spectra/
│   ├── teff_5500_logg_4.5_feh_0.0/
│   │   ├── spectrum_00001.fits
│   │   ├── spectrum_00002.fits
│   │   └── ...
│   └── ...
├── catalog.fits        # Master catalog with all parameters
└── metadata.json       # Library metadata
```

## Utilities

### Wavelength Operations

```python
from lvmstars.utils import (
    wavelength_to_velocity,
    velocity_to_wavelength,
    resample_spectrum,
    create_wavelength_grid
)

# Convert wavelength to velocity
velocity = wavelength_to_velocity(wavelength, rest_wavelength=5000.0)

# Resample spectrum
new_wavelength = create_wavelength_grid(4000, 7000, resolution=2000)
new_flux, new_error = resample_spectrum(wavelength, flux, new_wavelength, flux_error)
```

### Quality Checks

```python
from lvmstars.utils import mask_bad_pixels, calculate_snr

# Mask bad pixels
good_mask = mask_bad_pixels(flux, flux_error, threshold=5.0)
clean_flux = flux[good_mask]

# Calculate SNR in specific range
snr = calculate_snr(flux, flux_error, wavelength_range=(5000, 5500), wavelength=wavelength)
```

## References

- **GAIA**: ESA Gaia mission (https://www.cosmos.esa.int/gaia)
- **LVM**: Local Volume Mapper (https://www.sdss.org/surveys/lvm/)
- **Extinction Laws**:
  - Cardelli, Clayton, & Mathis 1989 (CCM89)
  - O'Donnell 1994
  - Calzetti et al. 2000
  - Fitzpatrick 1999

## Support

For issues or questions:
- GitHub Issues: https://github.com/sfsanchez72/LVMstars/issues
- Documentation: See README.md and example scripts
