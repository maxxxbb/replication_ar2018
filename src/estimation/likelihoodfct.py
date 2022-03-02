### likelihood function using estimagic
import numpy as np
import pandas as pd
from estimagic import estimate_ml
from scipy.stats import norm

# the function negloglike computes the negative of the log likelihood of observing our data given the parameters of the model.

# parameters:

# beta is the present bias parameter
# betahat is the perceived present bias parameter
# delta is the usual time-discounting parameter
# gamma and phi are the two parameters controlling the cost of effort function
# alpha is the projection bias parameter
# sigma is the standard deviation of the normal error term ϵ

# args:

# netdistance is (T-k)-(t-k) = T-t, the difference between the payment date T and the work time t
# wage is the amount paid per task in a certain session
# today is a dummy variable equal to one if the decision involves the choice of work today
# prediction is a dummy variable equal to one if the decision involves the choice of work in the future
# pb is a dummy equal to one if the subject completed 10 mandatory tasks on subject-day
# effort is the number of tasks completed by a subject in a session. It can range from a minimum of 10 to a maximum of 110
# ind_effort10 is a dummy equal to one if the subject's effort was equal to 10
# ind_effort110 is a dummy equal to one if the subject's effort was equal to 110


def estimate_table(args):
    """
    Args:
        data(Pd.DataFrame): DataFrame with observations
    Returns:
        ml_estimates(Pd.Dataframe): Table of primary structural estimates

    """
    params = start_params()
    res = estimate_ml(
        loglike=loglike,
        params=params,
        optimize_options={"algorithm": "scipy_neldermead"},
        loglike_kwargs={"args": args},
    )
    ml_estimates = res["summary_jacobian"].round(3)
    return ml_estimates


def loglike(params, args):

    """

    Args:
        params (pd.DataFrame): starting Parameters for Optimization
        data (pd.DataFrame):
    Returns:
        Dictionary with

    """

    predchoice = (
        (
            params.loc["phi", "value"]
            * (params.loc["delta", "value"] ** args["netdistance"])
            * (params.loc["beta", "value"] ** args["today"])
            * (params.loc["betahat", "value"] ** args["prediction"])
            * args["wage"]
        )
        ** (1 / (params.loc["gamma", "value"] - 1))
    ) - args["pb"] * params.loc["alpha", "value"]
    prob = (
        (1 - args["ind_effort10"] - args["ind_effort110"])
        * norm.pdf(args["effort"], predchoice, params.loc["sigma", "value"])
        + args["ind_effort10"]
        * (1 - norm.cdf((predchoice - args["effort"]) / params.loc["sigma", "value"]))
        + args["ind_effort110"]
        * norm.cdf((predchoice - args["effort"]) / params.loc["sigma", "value"])
    )
    index_p0 = [i for i in range(0, len(prob)) if prob[i] == 0]
    index_p1 = [i for i in range(0, len(prob)) if prob[i] == 1]

    # use a for loop to change the values   # change this to list comprehension
    for i in index_p0:
        prob[i] = 1e-4
    for i in index_p1:
        prob[i] = 1 - 1e-4

    contr = np.log(prob)

    return {"contributions": contr, "value": np.sum(contr)}


def start_params():
    """
    Define initial guesses Consistent with the ones used by Augenblick & Rabin in original paper
    and Pozzi & Nunnari in their replication

    """
    parm = [0.8, 1, 1, 2, 500, 7, 40]
    init_parm = pd.DataFrame(
        parm,
        columns=["value"],
        index=["beta", "betahat", "delta", "gamma", "phi", "alpha", "sigma"],
    )
    return init_parm


def load_args(data):

    """

    Args:
        data (pd.DataFrame): Dataset containing observations for 71 individuals

    Returns:
        args (pd.DataFrame): Arguments(Necessary Columns) for LL-Function

    """
    netdistance = np.array(data["netdistance"])
    wage = np.array(data["wage"])
    today = np.array(data["today"])
    prediction = np.array(data["prediction"])
    pb = np.array(data["pb"])
    effort = np.array(data["effort"])
    ind_effort10 = np.array(data["ind_effort10"])
    ind_effort110 = np.array(data["ind_effort110"])

    args = pd.DataFrame(
        {
            "netdistance": netdistance,
            "wage": wage,
            "today": today,
            "prediction": prediction,
            "pb": pb,
            "effort": effort,
            "ind_effort10": ind_effort10,
            "ind_effort110": ind_effort110,
        }
    )

    return args
