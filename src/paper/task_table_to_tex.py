import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC

tables = ["table1", "table2", "table1_full"]


@pytask.mark.parametrize(
    "depends_on, produces",
    [
        (BLD / "tables" / f"{table}.csv", SRC / "paper" / "tables" / f"{table}.tex")
        for table in tables
    ],
)
def task_convert_tables(depends_on, produces):
    """
    Converts tables (csv) into latex tabular
    """
    table = pd.read_csv(depends_on)
    with open(produces, "w") as tf:
        tf.write(table.to_latex(na_rep="-", index=False))
