import pickle

import numpy as np
import pandas as pd
import pytask
from scipy.stats import pearsonr as pear

from src.config import BLD


def get_column(res, spec):
    if spec != 4:
        param = pd.DataFrame(
            res, columns=["beta", "betahat", "delta", "gamma", "phi", "sigma"]
        )
        column = np.round(
            [
                np.mean(param["beta"]),
                np.median(param["beta"]),
                np.std(param["beta"]),
                np.mean(param["betahat"]),
                np.median(param["betahat"]),
                np.std(param["betahat"]),
                np.mean(param["delta"]),
                np.median(param["delta"]),
                np.std(param["delta"]),
                np.mean(param["gamma"]),
                np.median(param["gamma"]),
                np.std(param["gamma"]),
                np.nan,
                np.nan,
                np.nan,
                np.round(np.mean(param["beta"] < 1), 2),
                np.round(np.mean(param["betahat"] < 1), 2),
                np.corrcoef(param["beta"], param["betahat"])[0, 1],
                pear(param["beta"], param["betahat"])[1],
                int(len(param["beta"])),
            ],
            3,
        )
    else:
        param = pd.DataFrame(
            res, columns=["beta", "betahat", "delta", "gamma", "phi", "alpha", "sigma"]
        )
        column = np.round(
            [
                np.mean(param["beta"]),
                np.median(param["beta"]),
                np.std(param["beta"]),
                np.mean(param["betahat"]),
                np.median(param["betahat"]),
                np.std(param["betahat"]),
                np.mean(param["delta"]),
                np.median(param["delta"]),
                np.std(param["delta"]),
                np.mean(param["gamma"]),
                np.median(param["gamma"]),
                np.std(param["gamma"]),
                np.mean(param["alpha"]),
                np.median(param["alpha"]),
                np.std(param["alpha"]),
                np.round(np.mean(param["beta"] < 1), 2),
                np.round(np.mean(param["betahat"] < 1), 2),
                np.corrcoef(param["beta"], param["betahat"])[0, 1],
                pear(param["beta"], param["betahat"])[1],
                int(len(param["beta"])),
            ],
            3,
        )

    return column


def get_table2(res):
    rownames = [
        "mean(beta)",
        "median(beta)",
        "sd(beta)",
        "mean(beta_h)",
        "median(beta_h)",
        "sd(beta_h)",
        "mean(delta)",
        "median(delta)",
        "sd(delta)",
        "mean(gamma)",
        "median(gamma)",
        "sd(gamma)",
        "mean(alpha)",
        "median(alpha)",
        "sd(alpha)",
        "P[beta]<1",
        "P[beta_h]<1",
        "r(beta,beta_h)",
        "p-value r(beta,beta_h)",
        "Observations",
    ]
    table2 = pd.DataFrame(
        {
            "Primary Estimation": get_column(res[0], 1),
            "Early Decisions": get_column(res[1], 2),
            "Later Decisions": get_column(res[2], 3),
            "Proj. Bias": get_column(res[3], 4),
        },
        index=rownames,
    )
    return table2


@pytask.mark.depends_on(BLD / "estimation" / "individual_estimates.pkl")
@pytask.mark.produces(BLD / "tables" / "table2.csv")
def task_gettable2(depends_on, produces):
    with open(depends_on, "rb") as f:
        res = pickle.load(f)
    table2 = get_table2(res)
    table2.to_csv(produces)
