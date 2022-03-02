import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.config import SRC
from src.data_management.prepare_data import prepare_data


@pytest.fixture
def data():
    out = pd.read_stata(
        SRC / "replication_files" / "original_data" / "decisions_data.dta"
    )
    return out


@pytest.fixture
def ind_to_keep():
    out = pd.read_csv(SRC / "replication_files" / "original_data" / "ind_to_keep.csv")
    return out


def test_data_management(data, ind_to_keep):

    expected = mimic_replication_datamanagement(data, ind_to_keep)
    actual = prepare_data(data, ind_to_keep, full_dataset=False)
    assert_frame_equal(expected, actual)


def mimic_replication_datamanagement(dt, ind_keep):
    dt = dt[dt.wid.isin(ind_keep.wid_col1)]
    dt = dt[dt.bonusoffered != 1]
    dt["pb"] = dt["workdone1"] / 10
    dt["ind_effort10"] = (dt["effort"] == 10).astype(int)
    dt["ind_effort110"] = (dt["effort"] == 110).astype(int)
    dt.index = np.arange(len(dt.wid))
    return dt
