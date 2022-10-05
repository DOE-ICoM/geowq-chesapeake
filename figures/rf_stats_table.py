import sys
import pandas as pd
from tabulate import tabulate

sys.path.append(".")
from src import utils

# --- rf params table
dt = pd.read_csv("data/best_params_rf.csv")
dt = dt.drop(columns=["bootstrap", "min_samples_split"])
variables = dt.pop("variable")
dt.insert(0, "variable", variables)
utils.tabulate_to_latex(
    tabulate(dt.values, headers=[x for x in dt.columns], tablefmt="latex"),
    "figures/rf_stats_table_0_.tex",
    "Random Forest tuning results.",
    2,
)

# --- rf stats table
dt = pd.read_csv("data/rmse_rf.csv").round(2)
dt = dt.T.reset_index().rename(columns={"index": "variable", 0: "rmse"})
col_names = dt["variable"].copy()
dt = dt.merge(pd.read_csv("data/r2.csv"), on="variable").T[1:3].reset_index()
dt.columns = ["metric"] + [x for x in col_names]
headers = [x for x in dt.columns]
headers[0] = ""
dt["metric"] = ["RMSE", "$R^2$"]

tblate = tabulate(dt.values, headers=[x for x in dt.columns], tablefmt="latex_raw")

utils.tabulate_to_latex(
    tblate,
    "figures/rf_stats_table_1_.tex",
    "Random Forest model fit and predictive accuracy metrics. (RMSE = root mean square error, $R^2$ = coefficient of determination)",
    0,
)
