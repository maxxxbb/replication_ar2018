import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def prepare_data():
    """
    Loads dta-file into a Pandas Dataframe and drops individuals for which individual parameters
    could not be estimated ( total of 28 participants). Called by Pytask

    Returns:
        prepared_data (Pandas Dataframe)
    """
    data = pd.read_stata(SRC / "original_data" / "decisions_data.dta")
    ind = pd.read_csv(SRC / "original_data" / "ind_to_keep.csv")
    data = data[data.wid.isin(ind.wid_col1)]  # drops all individuals not in 'ind'
    data = data[data.bonusoffered != 1]  # remove observations where a bonus was offered
    data["pb"] = data["workdone1"] / 10
    # workdone1 can either be 10 or 0: dividing the variable by 10 creates the dummy
    data["ind_effort10"] = (data["effort"] == 10).astype(int)  # ind_effort10 dummy
    data["ind_effort110"] = (data["effort"] == 110).astype(int)  # ind_effort110 dummy
    data.index = np.arange(len(data.wid))

    return data


def save_data(data, path):
    """
    Serializes Dataframe
    """
    data = pd.DataFrame(data)
    data.to_pickle(path)


@pytask.mark.produces(BLD / "data" / "prepared_data.pkl")
def task_get_prepared_data(produces):
    prepared_data = prepare_data()
    save_data(prepared_data, produces)
