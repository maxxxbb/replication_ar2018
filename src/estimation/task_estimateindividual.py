import pickle

import pandas as pd
import pytask

from src.config import BLD
from src.estimation.estimation import estimate_table2


@pytask.mark.depends_on(BLD / "data" / "prepared_data_full.pkl")
@pytask.mark.produces(BLD / "estimation" / "individual_estimates.pkl")
def task_getestimates(depends_on, produces):
    data = pd.read_pickle(depends_on)
    results = [estimate_table2(data, spec) for spec in range(1, 5)]

    with open(produces, "wb") as f:
        pickle.dump(results, f)
