"""
GAIA catalog cross-matching functionality.
"""

from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from typing import Optional, List
import warnings


class GaiaMatch:
    """
    Represents a GAIA catalog match for a stellar spectrum.

    Attributes
    ----------
    source_id : int
        GAIA source identifier
    separation : float
        Angular separation in arcseconds
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    parallax : float
        Parallax in mas
    pmra : float
        Proper motion in RA in mas/yr
    pmdec : float
        Proper motion in Dec in mas/yr
    phot_g_mean_mag : float
        G-band mean magnitude
    teff : float
        Effective temperature in K (if available)
    logg : float
        Surface gravity (if available)
    feh : float
        Metallicity [Fe/H] (if available)
    alpha_fe : float
        Alpha enhancement (if available)
    n_neighbors : int
        Number of nearby sources
    """

    def __init__(self, gaia_data: dict, separation: float, n_neighbors: int = 0):
        """
        Initialize a GaiaMatch object.

        Parameters
        ----------
        gaia_data : dict
            Dictionary with GAIA catalog data
        separation : float
            Angular separation in arcseconds
        n_neighbors : int
            Number of nearby sources within search radius
        """
        self.source_id = gaia_data.get("source_id")
        self.separation = separation
        self.ra = gaia_data.get("ra")
        self.dec = gaia_data.get("dec")
        self.parallax = gaia_data.get("parallax")
        self.pmra = gaia_data.get("pmra")
        self.pmdec = gaia_data.get("pmdec")
        self.phot_g_mean_mag = gaia_data.get("phot_g_mean_mag")

        # Stellar parameters (may not be available for all sources)
        self.teff = gaia_data.get("teff_gspphot")
        self.logg = gaia_data.get("logg_gspphot")
        self.feh = gaia_data.get("mh_gspphot")
        self.alpha_fe = gaia_data.get("alphafe_gspphot")

        self.n_neighbors = n_neighbors
        self._gaia_data = gaia_data

    def is_single_star(self, max_neighbors: int = 0, max_separation: float = 1.0) -> bool:
        """
        Check if this is likely a single star.

        Parameters
        ----------
        max_neighbors : int
            Maximum number of allowed neighbors
        max_separation : float
            Maximum allowed separation in arcseconds

        Returns
        -------
        bool
            True if likely a single star
        """
        return (
            self.n_neighbors <= max_neighbors
            and self.separation <= max_separation
            and self.parallax is not None
            and self.parallax > 0
        )

    def has_parameters(self) -> bool:
        """
        Check if stellar parameters are available.

        Returns
        -------
        bool
            True if parameters are available
        """
        return self.teff is not None and self.logg is not None and self.feh is not None

    def __repr__(self):
        return (
            f"GaiaMatch(source_id={self.source_id}, "
            f'sep={self.separation:.2f}", n_neighbors={self.n_neighbors})'
        )


class GaiaCrossmatcher:
    """
    Cross-match LVM spaxels with GAIA catalog.
    """

    def __init__(self, gaia_release: str = "gaiadr3"):
        """
        Initialize the GAIA cross-matcher.

        Parameters
        ----------
        gaia_release : str
            GAIA data release to use (default: 'gaiadr3')
        """
        self.gaia_release = gaia_release
        Gaia.MAIN_GAIA_TABLE = gaia_release + ".gaia_source"

    def match_spaxel(
        self,
        coordinates: SkyCoord,
        radius: float = 1.0,
        max_sources: int = 10,
    ) -> Optional[GaiaMatch]:
        """
        Cross-match a spaxel position with GAIA catalog.

        Parameters
        ----------
        coordinates : SkyCoord
            Sky coordinates to match
        radius : float
            Search radius in arcseconds
        max_sources : int
            Maximum number of sources to return

        Returns
        -------
        GaiaMatch or None
            Best GAIA match if found, None otherwise
        """
        try:
            # Query GAIA around the coordinates
            query = f"""
            SELECT TOP {max_sources}
                source_id, ra, dec, parallax, pmra, pmdec,
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                teff_gspphot, logg_gspphot, mh_gspphot, alphafe_gspphot,
                DISTANCE(
                    POINT({coordinates.ra.deg}, {coordinates.dec.deg}),
                    POINT(ra, dec)
                ) AS separation
            FROM {self.gaia_release}.gaia_source
            WHERE 1=CONTAINS(
                POINT({coordinates.ra.deg}, {coordinates.dec.deg}),
                CIRCLE(ra, dec, {radius/3600.0})
            )
            ORDER BY separation ASC
            """

            job = Gaia.launch_job(query)
            results = job.get_results()

            if len(results) == 0:
                warnings.warn(f"No GAIA sources found within {radius} arcsec")
                return None

            # Get the closest match
            closest = results[0]
            gaia_data = {key: closest[key] for key in closest.colnames}

            # Calculate separation in arcseconds
            separation = closest["separation"] * 3600.0  # Convert from degrees

            # Count neighbors (excluding the closest source)
            n_neighbors = len(results) - 1

            return GaiaMatch(gaia_data, separation, n_neighbors)

        except Exception as e:
            warnings.warn(f"Error querying GAIA: {e}")
            return None

    def match_multiple_spaxels(
        self,
        coordinates_list: List[SkyCoord],
        radius: float = 1.0,
    ) -> List[Optional[GaiaMatch]]:
        """
        Cross-match multiple spaxels with GAIA catalog.

        Parameters
        ----------
        coordinates_list : list of SkyCoord
            List of sky coordinates to match
        radius : float
            Search radius in arcseconds

        Returns
        -------
        list of GaiaMatch or None
            List of GAIA matches (None for no match)
        """
        matches = []
        for coords in coordinates_list:
            match = self.match_spaxel(coords, radius=radius)
            matches.append(match)
        return matches

    def query_region(
        self,
        center: SkyCoord,
        radius: float = 60.0,
        magnitude_limit: Optional[float] = None,
    ) -> List[dict]:
        """
        Query all GAIA sources in a region.

        Parameters
        ----------
        center : SkyCoord
            Center of the region
        radius : float
            Search radius in arcseconds
        magnitude_limit : float, optional
            Limiting G magnitude

        Returns
        -------
        list of dict
            List of GAIA sources
        """
        try:
            mag_condition = ""
            if magnitude_limit is not None:
                mag_condition = f"AND phot_g_mean_mag < {magnitude_limit}"

            query = f"""
            SELECT
                source_id, ra, dec, parallax, pmra, pmdec,
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                teff_gspphot, logg_gspphot, mh_gspphot, alphafe_gspphot
            FROM {self.gaia_release}.gaia_source
            WHERE 1=CONTAINS(
                POINT({center.ra.deg}, {center.dec.deg}),
                CIRCLE(ra, dec, {radius/3600.0})
            )
            {mag_condition}
            """

            job = Gaia.launch_job(query)
            results = job.get_results()

            # Convert to list of dictionaries
            sources = []
            for row in results:
                source = {key: row[key] for key in row.colnames}
                sources.append(source)

            return sources

        except Exception as e:
            warnings.warn(f"Error querying GAIA region: {e}")
            return []
