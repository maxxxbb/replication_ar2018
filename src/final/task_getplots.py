import pickle

import pandas as pd
import plotly.express as px
import pytask

from src.config import BLD


@pytask.mark.depends_on(BLD / "estimation" / "individual_estimates.pkl")
@pytask.mark.produces(
    [
        BLD / "figures" / "figure1.png",
        BLD / "figures" / "figure2.png",
        BLD / "figures" / "figure3.png",
        BLD / "figures" / "figure4.png",
    ]
)
def task_create_plots(depends_on, produces):

    with open(depends_on, "rb") as f:
        estimates = pickle.load(f)[1]

    cols = ["beta", "betahat", "delta", "gamma"]
    df = pd.DataFrame(
        estimates, columns=["beta", "betahat", "delta", "gamma", "phi", "sigma"]
    )
    fig = [
        px.histogram(
            df,
            x=col,
            labels={col: "Value of " + col},
            marginal="box",
            template="ggplot2",
            nbins=30,
            width=300,
            height=300,
        )
        for col in cols
    ]
    [fig[i].write_image(produces[i]) for i in range(len(fig))]
