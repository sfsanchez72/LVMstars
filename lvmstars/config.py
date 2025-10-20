"""
Configuration management for LVM stellar library processing.
"""

import json
from typing import Dict


class Config:
    """
    Configuration manager for LVM stellar library.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize configuration.

        Parameters
        ----------
        config_file : str, optional
            Path to configuration JSON file
        """
        # Default configuration
        self.config = {
            "gaia": {
                "release": "gaiadr3",
                "search_radius": 1.0,  # arcseconds
                "max_neighbors": 0,  # for single star identification
            },
            "sky_subtraction": {
                "method": "median",  # median, mean, or weighted
                "outlier_sigma": 3.0,
                "n_nearby_spaxels": 5,
            },
            "dust_correction": {
                "extinction_law": "ccm89",  # ccm89, odonnell94, calzetti00, fitzpatrick99
                "rv": 3.1,
            },
            "wavelength": {
                "min": 3600.0,  # Angstroms
                "max": 9800.0,  # Angstroms
                "resolution": 2000.0,
            },
            "quality": {
                "min_snr": 5.0,
                "max_separation": 1.0,  # arcseconds for GAIA match
            },
            "output": {
                "directory": "./stellar_library",
                "format": "fits",
                "include_metadata": True,
            },
        }

        # Load configuration from file if provided
        if config_file:
            self.load(config_file)

    def load(self, config_file: str):
        """
        Load configuration from JSON file.

        Parameters
        ----------
        config_file : str
            Path to configuration file
        """
        with open(config_file, "r") as f:
            user_config = json.load(f)

        # Update configuration with user settings
        self._update_dict(self.config, user_config)

    def save(self, config_file: str):
        """
        Save configuration to JSON file.

        Parameters
        ----------
        config_file : str
            Path to output file
        """
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def _update_dict(self, d: Dict, u: Dict):
        """Recursively update dictionary d with values from u."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, *keys):
        """
        Get configuration value by key path.

        Parameters
        ----------
        *keys : str
            Key path (e.g., 'gaia', 'release')

        Returns
        -------
        Any
            Configuration value
        """
        value = self.config
        for key in keys:
            value = value[key]
        return value

    def set(self, *keys, value):
        """
        Set configuration value by key path.

        Parameters
        ----------
        *keys : str
            Key path
        value : Any
            Value to set
        """
        d = self.config
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    def __repr__(self):
        return f"Config({json.dumps(self.config, indent=2)})"


def create_default_config(output_file: str = "config.json"):
    """
    Create a default configuration file.

    Parameters
    ----------
    output_file : str
        Path to output configuration file
    """
    config = Config()
    config.save(output_file)
    print(f"Default configuration saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Creating default configuration...")
    config = Config()

    print("\nDefault GAIA settings:")
    print(f"  Release: {config.get('gaia', 'release')}")
    print(f"  Search radius: {config.get('gaia', 'search_radius')} arcsec")

    print("\nDefault dust correction settings:")
    print(f"  Extinction law: {config.get('dust_correction', 'extinction_law')}")
    print(f"  R_V: {config.get('dust_correction', 'rv')}")

    # Save example configuration
    output_file = "/tmp/lvmstars_config.json"
    config.save(output_file)
    print(f"\nConfiguration saved to {output_file}")

    # Test loading
    config2 = Config(output_file)
    print(f"\nConfiguration loaded successfully from {output_file}")
