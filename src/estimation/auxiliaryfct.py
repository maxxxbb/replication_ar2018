import numpy as np
import pandas as pd

from src.config import SRC


def start_params(spec):
    """
    Define initial guesses Consistent with the ones used by Augenblick & Rabin
    in their Stata code and Pozzi & Nunnari in their replication.
    For table 1 and column 4 of table 2 the additional
    parameter "alpha" will be estimated.

    Args:
        spec(int): specification parameter (specifies row of final table2)

    Returns:
        init_parm(pd.DataFrame): Contains initial guesses for optimization

    """
    if spec != 4:
        parm = [1, 1, 1, 2, 250, 50]
        init_parm = pd.DataFrame(
            parm,
            columns=["value"],
            index=["beta", "betahat", "delta", "gamma", "phi", "sigma"],
        )
    else:
        parm = [0.8, 1, 1, 2, 500, 7, 40]
        init_parm = pd.DataFrame(
            parm,
            columns=["value"],
            index=["beta", "betahat", "delta", "gamma", "phi", "alpha", "sigma"],
        )
    return init_parm


def load_args(data):

    """
    Loads necessary arguments for Maximum-Likelihood Estimation

    Args:
        data(pd.DataFrame): Dataset containing all observations for 71 individuals

    Returns:
        args(pd.DataFrame): Arguments(Necessary Columns) for ML-Estimations

    """
    arglist = [
        "netdistance",
        "wage",
        "today",
        "prediction",
        "pb",
        "effort",
        "ind_effort10",
        "ind_effort110",
    ]
    args = [np.array(data[arg]) for arg in arglist]

    args = pd.DataFrame({k: v for k, v in zip(arglist, args)})

    return args


def prepare_data_fortable2(data, spec):
    """
    Drops individuals when in list of individuals which were not considered
    in the paper for the respective specification due to computational issues
    in Stata.

    Args:
        data(Pd.DataFrame): full prepared dataset
        spec(int): specification parameter

    Returns:
        out(Pd.DataFrame): Dataset with individuals kept in each specification

    """
    ind = pd.read_csv(SRC / "replication_files" / "original_data" / "ind_to_keep.csv")
    if spec == 1:
        out = data[data.wid.isin(ind.wid_col1)]
    elif spec == 2:
        out = data[data.wid.isin(ind.wid_col2)]
    elif spec == 3:
        out = data[data.wid.isin(ind.wid_col3)]
    elif spec == 4:
        out = data[data.wid.isin(ind.wid_col4)]
    return out


def getind(dataset, wid, spec):
    """
    Specifies individual dataset for column 2 and 3 of table 2:
    In column 2 early decisions are considered while in column 3
    late decisions are considered. For col 1 and 4 all decicions
    are taken into account.

    Args:
        dataset(Pd.DataFrame): dataset containing all considered individuals
        wid(int): Individual ID
        spec(int): specification parameter
    Returns:
        dataset_ind(Pd.DataFrame):  Relevant individual-level dataframe
            containing only observations for individual whose ID=wid.

    """
    dataset_ind = dataset[dataset.wid == wid]
    if spec == 2:
        dataset_ind = dataset_ind[dataset_ind.decisiondatenum < 4]
    elif spec == 3:
        dataset_ind = dataset_ind[dataset_ind.decisiondatenum >= 4]
    return dataset_ind
