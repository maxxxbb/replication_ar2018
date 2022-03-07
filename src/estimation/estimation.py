import numpy as np
from estimagic import estimate_ml
from estimagic import maximize

from src.estimation.auxiliaryfct import getind
from src.estimation.auxiliaryfct import load_args
from src.estimation.auxiliaryfct import prepare_data_fortable2
from src.estimation.auxiliaryfct import start_params
from src.estimation.likelihoodfct import loglike


def estimate_table2(dataset, spec):
    data = prepare_data_fortable2(dataset, spec)
    params = start_params(spec)
    results = [
        estimate_individual_ll(wid, data, spec, params) for wid in np.unique(data.wid)
    ]
    return results


def estimate_individual_ll(wid, dataset, spec, params):

    dt_ind = getind(dataset, wid, spec)
    args_ind = load_args(dt_ind)
    sol = maximize(
        criterion=loglike,
        params=params,
        algorithm="scipy_neldermead",
        criterion_kwargs={"args": args_ind, "spec": spec},
        algo_options={"stopping_max_iterations": 2500},
    )
    return sol["solution_x"]


def estimate_table1(args):
    """
    Args:
        data(Pd.DataFrame): DataFrame with observations
    Returns:
        ml_estimates(Pd.Dataframe): Table of primary structural estimates

    """
    params = start_params(4)
    res = estimate_ml(
        loglike=loglike,
        params=params,
        optimize_options={"algorithm": "scipy_neldermead"},
        loglike_kwargs={"args": args, "spec": 4},
    )
    ml_estimates = res["summary_jacobian"].round(3)
    return ml_estimates
