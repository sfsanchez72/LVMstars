"""
Tests for the parameters module.
"""

from lvmstars.parameters import StellarParameters, ParameterGrid


def test_stellar_parameters_creation():
    """Test basic StellarParameters creation."""
    params = StellarParameters(teff=5800, logg=4.5, feh=0.0, alpha_fe=0.1)

    assert params.teff == 5800
    assert params.logg == 4.5
    assert params.feh == 0.0
    assert params.alpha_fe == 0.1


def test_stellar_parameters_validation():
    """Test parameter validation."""
    # Valid parameters
    params = StellarParameters(teff=5800, logg=4.5, feh=0.0)
    assert params.validate() is True

    # Invalid temperature
    params_invalid = StellarParameters(teff=100000, logg=4.5, feh=0.0)
    assert params_invalid.validate() is False


def test_is_complete():
    """Test completeness check."""
    # Complete parameters
    params = StellarParameters(teff=5800, logg=4.5, feh=0.0)
    assert params.is_complete() is True

    # Incomplete parameters
    params_incomplete = StellarParameters(teff=5800, logg=4.5)
    assert params_incomplete.is_complete() is False


def test_to_dict():
    """Test conversion to dictionary."""
    params = StellarParameters(teff=5800, logg=4.5, feh=0.0, alpha_fe=0.1)
    data = params.to_dict()

    assert data["teff"] == 5800
    assert data["logg"] == 4.5
    assert data["feh"] == 0.0
    assert data["alpha_fe"] == 0.1


def test_from_dict():
    """Test creation from dictionary."""
    data = {"teff": 5800, "logg": 4.5, "feh": 0.0, "alpha_fe": 0.1}
    params = StellarParameters.from_dict(data)

    assert params.teff == 5800
    assert params.logg == 4.5
    assert params.feh == 0.0
    assert params.alpha_fe == 0.1


def test_spectral_type_estimate():
    """Test spectral type estimation."""
    # G-type star
    params_g = StellarParameters(teff=5800, logg=4.5)
    assert params_g.spectral_type_estimate() == "GV"

    # K-type giant
    params_k = StellarParameters(teff=4500, logg=2.0)
    assert params_k.spectral_type_estimate() == "KIII"

    # M-type dwarf
    params_m = StellarParameters(teff=3500, logg=4.8)
    assert params_m.spectral_type_estimate() == "MV"


def test_parameter_grid_creation():
    """Test ParameterGrid creation."""
    grid = ParameterGrid(
        teff_range=(4000, 7000),
        teff_step=500,
        logg_range=(3.0, 5.0),
        logg_step=0.5,
        feh_range=(-1.0, 0.5),
        feh_step=0.5,
    )

    assert len(grid.teff_grid) == 7  # 4000, 4500, ..., 7000
    assert len(grid.logg_grid) == 5  # 3.0, 3.5, 4.0, 4.5, 5.0
    assert len(grid.feh_grid) == 4  # -1.0, -0.5, 0.0, 0.5


def test_find_nearest_grid_point():
    """Test finding nearest grid point."""
    grid = ParameterGrid(
        teff_range=(4000, 7000),
        teff_step=500,
        logg_range=(3.0, 5.0),
        logg_step=0.5,
        feh_range=(-1.0, 0.5),
        feh_step=0.5,
    )

    nearest = grid.find_nearest_grid_point(teff=5823, logg=4.47, feh=-0.23)

    assert nearest["teff"] == 6000  # Nearest 500K grid point
    assert nearest["logg"] == 4.5  # Nearest 0.5 grid point
    assert nearest["feh"] == 0.0  # Nearest 0.5 grid point


def test_get_grid_size():
    """Test grid size calculation."""
    grid = ParameterGrid(
        teff_range=(4000, 6000),
        teff_step=1000,
        logg_range=(3.0, 5.0),
        logg_step=1.0,
        feh_range=(-1.0, 0.0),
        feh_step=0.5,
    )

    # 3 Teff × 3 logg × 3 [Fe/H] = 27 points
    assert grid.get_grid_size() == 27
