import numpy as np
from scipy.stats import norm


def loglike(params, args, spec):
    """
    Computes the log likelihood of observing our data given the parameters of the model.
        Args:
            - params(pd.DataFrame): Parameters to be estimated in Optimization
            - args(pd.DataFrame): Dataframe containing the arguments of the model

        Returns:
            - out(dict): Dictionary containing individual contributions("contr")
            and sum of loglikelihood-function("value")

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

    if spec == 4:
        beta, betahat, delta, gamma, phi, alpha, sigma = params["value"]
    else:
        beta, betahat, delta, gamma, phi, sigma = params["value"]

    netdistance, wage, today, prediction, pb, effort, ind_effort10, ind_effort110 = [
        args[i] for i in arglist
    ]
    # predicted choice from optimality condition of agent
    predchoice = (
        phi * (delta ** netdistance) * (beta ** today) * (betahat ** prediction) * wage
    ) ** (1 / (gamma - 1))

    if spec == 4:
        predchoice = predchoice - pb * alpha

    prob = (
        (1 - ind_effort10 - ind_effort110) * norm.pdf(effort, predchoice, sigma)
        + ind_effort10 * (1 - norm.cdf((predchoice - effort) / sigma))
        + ind_effort110 * norm.cdf((predchoice - effort) / sigma)
    )
    # if prob is zero or one add small value to avoid problems taking logs
    index_p0 = [i for i in range(0, len(prob)) if prob[i] == 0]
    index_p1 = [i for i in range(0, len(prob)) if prob[i] == 1]

    for i in index_p0:
        prob[i] = 1e-4

    for i in index_p1:
        prob[i] = 1 - 1e-4

    contr = np.log(prob)
    out = {"contributions": contr, "value": np.sum(contr)}
    return out
