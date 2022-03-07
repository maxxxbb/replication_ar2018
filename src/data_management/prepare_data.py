import numpy as np


def prepare_data(data, ind, full_dataset=False):
    """

    Prepares Data for use in Analysis.
    removes observatoins where bonus was offerde
    Args:
        data(Dta-File): Whole Dataset
        full_dataset(Boolean): indicates whether individuals should be left
        out of analysis or not
    Returns:
        prepared_data(Pandas Dataframe): Dataset for Analysis
    """
    data = data[data.bonusoffered != 1]
    data["pb"] = data["workdone1"] / 10
    data["ind_effort10"] = (data["effort"] == 10).astype(int)  # ind_effort10 dummy
    data["ind_effort110"] = (data["effort"] == 110).astype(int)  # ind_effort110 dummy

    if full_dataset == False:
        out = data[data.wid.isin(ind.wid_col1)]
    else:
        out = data

    out.index = np.arange(len(out.wid))

    return out
