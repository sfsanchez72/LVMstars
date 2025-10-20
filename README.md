# LVMstars

LVM Stellar Library - Building a stellar library from LVM observations

## Overview

This repository contains tools for building a stellar library from LVM (Local Volume Mapper) observations, matching its spectral resolution and coverage. The pipeline identifies single-star spectra through GAIA cross-matching, removes sky and ISM residuals, corrects for dust, and tags spectra with physical parameters.

## Features

- **GAIA Cross-matching**: Identifies single-star spectra in LVM spaxels
- **Sky/ISM Residual Subtraction**: Removes contamination from nearby spaxels
- **Dust Correction**: Applies corrections using DAP (Data Analysis Pipeline) results
- **Stellar Parameter Tagging**: Associates each spectrum with Teff, log g, [Fe/H], and [α/Fe]

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Workflow

```python
from lvmstars import StellarSpectrum, GaiaCrossmatcher, SkySubtractor, DustCorrector

# Load LVM spaxel data
spectrum = StellarSpectrum.from_lvm_spaxel(spaxel_data)

# Cross-match with GAIA to verify single star
matcher = GaiaCrossmatcher()
gaia_match = matcher.match_spaxel(spectrum.coordinates, radius=1.0)

if gaia_match.is_single_star():
    # Subtract sky/ISM residuals
    subtractor = SkySubtractor()
    cleaned_spectrum = subtractor.subtract_nearby_spaxels(spectrum, nearby_spaxels)
    
    # Apply dust correction
    corrector = DustCorrector()
    corrected_spectrum = corrector.correct(cleaned_spectrum, dap_results)
    
    # Tag with stellar parameters
    corrected_spectrum.set_parameters(
        teff=gaia_match.teff,
        logg=gaia_match.logg,
        feh=gaia_match.feh,
        alpha_fe=gaia_match.alpha_fe
    )
    
    # Save to library
    corrected_spectrum.save_to_library()
```

## Module Structure

- `lvmstars.spectrum`: Core spectral data structures
- `lvmstars.gaia`: GAIA catalog cross-matching functionality
- `lvmstars.subtraction`: Sky and ISM residual subtraction
- `lvmstars.dust`: Dust correction using DAP results
- `lvmstars.parameters`: Stellar parameter management
- `lvmstars.utils`: Utility functions

## Requirements

- Python >= 3.8
- numpy >= 1.20
- astropy >= 5.0
- scipy >= 1.7
- astroquery >= 0.4

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

BSD-3-Clause

## Citation

If you use this code in your research, please cite the LVM project.
