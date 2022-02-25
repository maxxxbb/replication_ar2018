import numpy as np


def prepare_data(data):
    """

    Prepares Data for use in Analysis
    Args:
        data(Dta-File): Whole Dataset

    Returns:
        prepared_data(Pandas Dataframe): Dataset for Analysis
    """
    data = data[data.bonusoffered != 1]  # remove observations where a bonus was offered
    data["pb"] = data["workdone1"] / 10
    data["ind_effort10"] = (data["effort"] == 10).astype(int)  # ind_effort10 dummy
    data["ind_effort110"] = (data["effort"] == 110).astype(int)  # ind_effort110 dummy
    data.index = np.arange(len(data.wid))

    return data


def drop_individuals(data, ind):
    """
    Drops individuals from Dataframe for which individual parameters
    could not be estimated ( total of 28 participants).
    Args:
        data(Dta-File): Whole Dataset
        ind(Dta-File): Individuals for which analysis was possible with Stata-Algorithm

    Returns:
        prepared_data(Pandas Dataframe): Dataset for Analysis
    """

    data = data[data.wid.isin(ind.wid_col1)]  # drops all individuals not in 'ind'
    return data
