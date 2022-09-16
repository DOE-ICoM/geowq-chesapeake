import sys
import pandas as pd
from tabulate import tabulate

sys.path.append(".")
from src import utils

dt = pd.read_csv("data/best_params_rf.csv")
dt = dt.drop(columns=["bootstrap", "min_samples_split"])
variables = dt.pop("variable")
dt.insert(0, "variable", variables)
utils.tabulate_to_latex(
    tabulate(
        dt.values, headers=[x for x in dt.columns], tablefmt="latex"
    ),
    "figures/rf_stats_table_0_.tex",
    "Random Forest tuning results.",
    2,
)

dt = pd.read_csv("data/rmse_rf.csv").round(2)
dt = dt.T.reset_index().rename(columns={"index":"variable", 0:"rmse"})
col_names = dt["variable"].copy()
dt = dt.merge(pd.read_csv("data/r2.csv"), on="variable").T[1:3].reset_index()
dt.columns = ["metric"] + [x for x in col_names]

utils.tabulate_to_latex(
    tabulate(
        dt.values, headers=[x for x in dt.columns], tablefmt="latex"
    ),
    "figures/rf_stats_table_1_.tex",
    "Random Forest fitting results. Values shown are root mean square error.",
    0,
)
