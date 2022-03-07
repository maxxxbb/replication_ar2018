import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.data_management.prepare_data import prepare_data


@pytask.mark.depends_on(
    [
        SRC / "replication_files" / "original_data" / "decisions_data.dta",
        SRC / "replication_files" / "original_data" / "ind_to_keep.csv",
    ]
)
@pytask.mark.produces(
    [
        BLD / "data" / "prepared_data_full.pkl",
        BLD / "data" / "prepared_data.pkl",
    ]
)
def task_get_prepared_data(depends_on, produces):
    data = pd.read_stata(depends_on[0])
    ind = pd.read_csv(depends_on[1])

    prepared_data_full = prepare_data(data, ind, full_dataset=True)
    prepared_data = prepare_data(data, ind)

    prepared_data_full.to_pickle(produces[0])
    prepared_data.to_pickle(produces[1])
