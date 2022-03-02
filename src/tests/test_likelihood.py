import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_series_equal

from src.config import BLD
from src.estimation.likelihoodfct import estimate_table
from src.estimation.likelihoodfct import load_args
from src.estimation.likelihoodfct import loglike
from src.estimation.likelihoodfct import start_params


@pytest.fixture
def args():
    data = pd.read_pickle(BLD / "data" / "prepared_data.pkl")
    out = load_args(data)
    return out


def test_ll(args):

    expected = -29152.287491937557
    params = start_params()
    actual = loglike(params, args)["value"]

    assert_almost_equal(actual, expected, decimal=5)


def test_estimagic(args):

    expected = pd.Series(
        [0.835, 0.999, 1.003, 2.145, 723.974, 7.307, 42.625],
        index=["beta", "betahat", "delta", "gamma", "phi", "alpha", "sigma"],
        name="value",
    )

    actual = estimate_table(args)["value"]

    assert_series_equal(expected, actual)
