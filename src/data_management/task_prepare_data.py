import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def prepare_data(data, ind):  # at a later point add option for full dataset
    """

    Loads dta-file into a Pandas Dataframe and drops individuals for which individual parameters
    could not be estimated ( total of 28 participants).
    Args:
        data(Dta-File): Whole Dataset
        ind(Dta-File): Individuals for which analysis was possible with Stata-Algorithm

    Returns:
        prepared_data(Pandas Dataframe): Dataset for Analysis
    """

    data = data[data.wid.isin(ind.wid_col1)]  # drops all individuals not in 'ind'
    data = data[data.bonusoffered != 1]  # remove observations where a bonus was offered
    data["pb"] = data["workdone1"] / 10
    # workdone1 can either be 10 or 0: dividing the variable by 10 creates the dummy
    data["ind_effort10"] = (data["effort"] == 10).astype(int)  # ind_effort10 dummy
    data["ind_effort110"] = (data["effort"] == 110).astype(int)  # ind_effort110 dummy
    data.index = np.arange(len(data.wid))

    return data


@pytask.mark.depends_on(
    [
        SRC / "Replication_Files" / "original_data" / "decisions_data.dta",
        SRC / "Replication_Files" / "original_data" / "ind_to_keep.csv",
    ]
)
@pytask.mark.produces(BLD / "data" / "prepared_data.pkl")
def task_get_prepared_data(depends_on, produces):
    data = pd.read_stata(depends_on[0])
    ind = pd.read_csv(depends_on[1])
    prepared_data = prepare_data(data, ind)
    prepared_data.to_pickle(produces)
