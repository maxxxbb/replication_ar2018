import pandas as pd
import pytask

from src.config import BLD
from src.estimation.auxiliaryfct import load_args
from src.estimation.estimation import estimagic_table1


@pytask.mark.depends_on(
    {
        "data": BLD / "data" / "prepared_data.pkl",
        "full_data": BLD / "data" / "prepared_data_full.pkl",
    }
)
@pytask.mark.produces(
    {
        "table1": BLD / "tables" / "table1.csv",
        "table1_full": BLD / "tables" / "table1_full.csv",
    }
)
def task_get_table_of_estimates(depends_on, produces):
    """
    Executes primary aggregate estimation and saves table
    containing estimates in a csv-file
    """
    data = pd.read_pickle(depends_on["data"])
    data_full = pd.read_pickle(depends_on["full_data"])
    args = load_args(data)
    args_full = load_args(data_full)
    ml_estimates = estimagic_table1(args)
    ml_estimates_full = estimagic_table1(args_full)
    ml_estimates.to_csv(produces["table1"])
    ml_estimates_full.to_csv(produces["table1_full"])
