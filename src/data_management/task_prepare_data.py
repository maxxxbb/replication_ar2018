import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    [
        SRC / "replication_files" / "original_data" / "decisions_data.dta",
        SRC / "replication_files" / "original_data" / "ind_to_keep.csv",
    ]
)
@pytask.mark.produces(
    [
        BLD / "data" / "prepared_data_full.pkl",
        BLD / "data" / "prepared_data.pkl",
    ]
)
def task_get_prepared_data(depends_on, produces):
    data = pd.read_stata(depends_on[0])
    ind = pd.read_csv(depends_on[1])

    prepared_data_full = prepare_data(data, ind, full_dataset=True)
    prepared_data = prepare_data(data, ind)

    prepared_data_full.to_pickle(produces[0])
    prepared_data.to_pickle(produces[1])


def prepare_data(data, ind, full_dataset=False):
    """

    Prepares Data for the Estimation. Removes observations where bonus was offered,
    adds dummy variables to dataset and optionally drops subjects whose
    individual estimates did not converge in Stata.

    Args:
        data(Dta-File): Raw Decisions-Dataset containing observations for 100 individuals.
        full_dataset(bool): Indicates whether subjects are dropped

    Returns:
        prepared_data(pd.DataFrame): Prepared data for ML-Analysis

    """
    data = data[data.bonusoffered != 1]

    pb = pd.Series(data["workdone1"] / 10, name="pb")
    ind_effort10 = pd.Series((data["effort"] == 10).astype(int), name="ind_effort10")
    ind_effort110 = pd.Series((data["effort"] == 110).astype(int), name="ind_effort110")

    df = pd.concat([data, pb, ind_effort10, ind_effort110], axis=1)

    if full_dataset is False:
        out = df[df.wid.isin(ind.wid_col1)]
    else:
        out = df

    out.index = np.arange(len(out.wid))

    return out
