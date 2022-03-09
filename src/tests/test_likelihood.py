import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_series_equal

from src.config import SRC
from src.data_management.task_prepare_data import prepare_data
from src.estimation.auxiliaryfct import load_args
from src.estimation.auxiliaryfct import start_params
from src.estimation.estimation import estimagic_table1
from src.estimation.estimation import estimate_individual_ll
from src.estimation.likelihoodfct import loglike


@pytest.fixture
def data():
    dt = pd.read_stata(
        SRC / "replication_files" / "original_data" / "decisions_data.dta"
    )
    ind = pd.read_csv(SRC / "replication_files" / "original_data" / "ind_to_keep.csv")
    data = prepare_data(dt, ind)
    return data


def test_ll(data):
    """
    Tests whether Loglikelihoodfunction is set up correctly.
    Fixture is from replication of Nunnari&Pozzi.
    """
    expected = -29152.287491937557
    args = load_args(data)
    params = start_params(spec=4)
    actual = loglike(params, args, spec=4)["value"]

    assert_almost_equal(actual, expected, decimal=5)


def test_estimate_ml(data):
    """
    Tests whether primary structural estimation yields correct results.
    Fixture is from replication of Nunnari&Pozzi.
    """
    args = load_args(data)
    expected = pd.Series(
        [0.835, 0.999, 1.003, 2.145, 723.974, 7.307, 42.625],
        index=["beta", "betahat", "delta", "gamma", "phi", "alpha", "sigma"],
        name="value",
    )
    actual = estimagic_table1(args)["value"]

    assert_series_equal(expected, actual)


def test_estimate_individual_ll(data):
    """
    Tests whether participant-specific parameters are estimated correctly.
    """
    params = start_params(spec=3)
    expected = np.array([0.809, 0.832, 1.063, 1.431, 53.219, 391.648])
    actual = np.round(estimate_individual_ll(3, data, 3, params), 3)
    assert_almost_equal(expected, actual, 3)
