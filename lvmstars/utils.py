"""
Utility functions for LVM stellar library processing.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import List, Optional, Tuple
import warnings


def calculate_distances(
    coord_list: List[SkyCoord],
    reference: SkyCoord,
) -> np.ndarray:
    """
    Calculate angular distances from a reference coordinate.

    Parameters
    ----------
    coord_list : list of SkyCoord
        List of coordinates
    reference : SkyCoord
        Reference coordinate

    Returns
    -------
    np.ndarray
        Angular distances in arcseconds
    """
    separations = []
    for coord in coord_list:
        sep = reference.separation(coord).to(u.arcsec).value
        separations.append(sep)
    return np.array(separations)


def find_nearby_spaxels(
    target_coord: SkyCoord,
    spaxel_coords: List[SkyCoord],
    radius: float,
    exclude_target: bool = True,
) -> List[int]:
    """
    Find indices of spaxels within a radius of target.

    Parameters
    ----------
    target_coord : SkyCoord
        Target coordinate
    spaxel_coords : list of SkyCoord
        List of all spaxel coordinates
    radius : float
        Search radius in arcseconds
    exclude_target : bool
        Exclude the target itself (for zero-distance matches)

    Returns
    -------
    list of int
        Indices of nearby spaxels
    """
    distances = calculate_distances(spaxel_coords, target_coord)

    if exclude_target:
        nearby = np.where((distances > 0) & (distances <= radius))[0]
    else:
        nearby = np.where(distances <= radius)[0]

    return nearby.tolist()


def wavelength_to_velocity(
    wavelength: np.ndarray,
    rest_wavelength: float,
) -> np.ndarray:
    """
    Convert wavelength to velocity (km/s).

    Parameters
    ----------
    wavelength : np.ndarray
        Observed wavelength array
    rest_wavelength : float
        Rest wavelength

    Returns
    -------
    np.ndarray
        Velocity array in km/s
    """
    c = 299792.458  # km/s
    velocity = c * (wavelength - rest_wavelength) / rest_wavelength
    return velocity


def velocity_to_wavelength(
    velocity: np.ndarray,
    rest_wavelength: float,
) -> np.ndarray:
    """
    Convert velocity (km/s) to wavelength.

    Parameters
    ----------
    velocity : np.ndarray
        Velocity array in km/s
    rest_wavelength : float
        Rest wavelength

    Returns
    -------
    np.ndarray
        Wavelength array
    """
    c = 299792.458  # km/s
    wavelength = rest_wavelength * (1 + velocity / c)
    return wavelength


def resample_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    new_wavelength: np.ndarray,
    flux_error: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Resample spectrum to a new wavelength grid using linear interpolation.

    Parameters
    ----------
    wavelength : np.ndarray
        Original wavelength array
    flux : np.ndarray
        Original flux array
    new_wavelength : np.ndarray
        New wavelength array
    flux_error : np.ndarray, optional
        Original flux error array

    Returns
    -------
    tuple
        (resampled_flux, resampled_error)
    """
    resampled_flux = np.interp(new_wavelength, wavelength, flux)

    resampled_error = None
    if flux_error is not None:
        # Simple error propagation (could be improved)
        resampled_error = np.interp(new_wavelength, wavelength, flux_error)

    return resampled_flux, resampled_error


def mask_bad_pixels(
    flux: np.ndarray,
    flux_error: np.ndarray,
    threshold: float = 5.0,
) -> np.ndarray:
    """
    Create a mask for bad pixels based on flux/error ratio.

    Parameters
    ----------
    flux : np.ndarray
        Flux array
    flux_error : np.ndarray
        Flux error array
    threshold : float
        SNR threshold below which pixels are masked

    Returns
    -------
    np.ndarray
        Boolean mask (True = good pixel)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(flux) / flux_error

    good_mask = np.isfinite(flux) & np.isfinite(flux_error) & (flux_error > 0) & (snr >= threshold)

    return good_mask


def estimate_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    window_size: int = 51,
    order: int = 3,
) -> np.ndarray:
    """
    Estimate continuum using median filtering and polynomial fitting.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array
    flux : np.ndarray
        Flux array
    window_size : int
        Size of median filter window (must be odd)
    order : int
        Polynomial order for fitting

    Returns
    -------
    np.ndarray
        Estimated continuum
    """
    from scipy.signal import medfilt
    from scipy.interpolate import UnivariateSpline

    # Apply median filter to remove narrow features
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd

    filtered = medfilt(flux, kernel_size=window_size)

    # Fit spline to filtered data
    try:
        spline = UnivariateSpline(wavelength, filtered, k=order, s=len(wavelength))
        continuum = spline(wavelength)
    except Exception:
        # Fall back to polynomial if spline fails
        coeffs = np.polyfit(wavelength, filtered, order)
        continuum = np.polyval(coeffs, wavelength)

    return continuum


def calculate_snr(
    flux: np.ndarray,
    flux_error: np.ndarray,
    wavelength_range: Optional[Tuple[float, float]] = None,
    wavelength: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate signal-to-noise ratio.

    Parameters
    ----------
    flux : np.ndarray
        Flux array
    flux_error : np.ndarray
        Flux error array
    wavelength_range : tuple, optional
        (min, max) wavelength range to compute SNR
    wavelength : np.ndarray, optional
        Wavelength array (required if wavelength_range is given)

    Returns
    -------
    float
        Median SNR
    """
    if wavelength_range is not None:
        if wavelength is None:
            raise ValueError("wavelength array required when wavelength_range is given")
        mask = (wavelength >= wavelength_range[0]) & (wavelength <= wavelength_range[1])
        flux = flux[mask]
        flux_error = flux_error[mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        snr = flux / flux_error
        median_snr = np.nanmedian(snr[np.isfinite(snr)])

    return median_snr if np.isfinite(median_snr) else 0.0


def create_wavelength_grid(
    wmin: float,
    wmax: float,
    resolution: Optional[float] = None,
    n_pixels: Optional[int] = None,
    log_scale: bool = False,
) -> np.ndarray:
    """
    Create a wavelength grid.

    Parameters
    ----------
    wmin : float
        Minimum wavelength
    wmax : float
        Maximum wavelength
    resolution : float, optional
        Spectral resolution (R = λ/Δλ)
    n_pixels : int, optional
        Number of pixels (alternative to resolution)
    log_scale : bool
        Use logarithmic spacing

    Returns
    -------
    np.ndarray
        Wavelength grid
    """
    if resolution is not None:
        # Calculate number of pixels from resolution
        if log_scale:
            # For log spacing: Δlog(λ) = 1 / (R * ln(10))
            delta_log = 1.0 / (resolution * np.log(10))
            log_wmin = np.log10(wmin)
            log_wmax = np.log10(wmax)
            n_pix = int((log_wmax - log_wmin) / delta_log) + 1
            wavelength = np.logspace(log_wmin, log_wmax, n_pix)
        else:
            # For linear spacing
            delta_wave = wmin / resolution  # Approximate
            n_pix = int((wmax - wmin) / delta_wave) + 1
            wavelength = np.linspace(wmin, wmax, n_pix)
    elif n_pixels is not None:
        if log_scale:
            wavelength = np.logspace(np.log10(wmin), np.log10(wmax), n_pixels)
        else:
            wavelength = np.linspace(wmin, wmax, n_pixels)
    else:
        raise ValueError("Either resolution or n_pixels must be provided")

    return wavelength
