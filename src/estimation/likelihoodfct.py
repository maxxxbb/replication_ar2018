import numpy as np
from scipy.stats import norm


def loglike(params, args, spec):
    """
    Args:
        params (pd.DataFrame): starting Parameters for Optimization
        data (pd.DataFrame):
    Returns:
        Dictionary with

    """
    predchoice = (
        params.loc["phi", "value"]
        * (params.loc["delta", "value"] ** args["netdistance"])
        * (params.loc["beta", "value"] ** args["today"])
        * (params.loc["betahat", "value"] ** args["prediction"])
        * args["wage"]
    ) ** (1 / (params.loc["gamma", "value"] - 1))

    if spec == 4:
        predchoice = predchoice - args["pb"] * params.loc["alpha", "value"]

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

    for i in index_p0:
        prob[i] = 1e-4

    for i in index_p1:
        prob[i] = 1 - 1e-4

    contr = np.log(prob)

    return {"contributions": contr, "value": np.sum(contr)}
