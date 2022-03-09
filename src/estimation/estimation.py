import numpy as np
from estimagic import estimate_ml
from estimagic import maximize

from src.estimation.auxiliaryfct import getind
from src.estimation.auxiliaryfct import load_args
from src.estimation.auxiliaryfct import prepare_data_fortable2
from src.estimation.auxiliaryfct import start_params
from src.estimation.likelihoodfct import loglike


def estimagic_table2(dataset, spec):
    """
    Estimates individual structural estimates for
    all considered participants and respective
    specification.

    Args:
        dataset(Pd.Dataframe): full prepared dataset
        spec(int): specification parameter
    Returns:
        results(list): list of arrays containing the subject
            specific paramater estimates

    """
    data = prepare_data_fortable2(dataset, spec)
    params = start_params(spec)
    results = [
        estimate_individual_ll(wid, data, spec, params) for wid in np.unique(data.wid)
    ]
    return results


def estimate_individual_ll(wid, dataset, spec, params):
    """
    Estimates individual structural estimate.

    Args:
        wid(int): participant index
        dataset(Pd.DataFrame): prepared dataframe with subjects to keep in each column.
        spec(int): specification parameter
        params(Pd.DataFrame): parameters to estimate
    Returns:
        out(np.Array): Array containing individual structural estimates

    """
    dt_ind = getind(dataset, wid, spec)
    args_ind = load_args(dt_ind)
    sol = maximize(
        criterion=loglike,
        params=params,
        algorithm="scipy_neldermead",
        criterion_kwargs={"args": args_ind, "spec": spec},
        algo_options={"stopping_max_iterations": 2500},
    )
    out = sol["solution_x"]
    return out


def estimagic_table1(args):
    """
    Primary aggregate structural estimation.

    Args:
        args(Pd.DataFrame): DataFrame containing args for Likelihood function
    Returns:
        ml_estimates(Pd.Dataframe): Table of primary structural estimates

    """
    params = start_params(4)
    res = estimate_ml(
        loglike=loglike,
        params=params,
        optimize_options={"algorithm": "scipy_neldermead"},
        loglike_kwargs={"args": args, "spec": 4},
    )  # tried "bhhh" here: invalid algorithm
    ml_estimates = res["summary_jacobian"].round(3)
    return ml_estimates
