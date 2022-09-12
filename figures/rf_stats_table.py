import sys
import pandas as pd
from tabulate import tabulate

sys.path.append(".")
from src import utils

dt = pd.read_csv("data/best_params_rf.csv")
utils.tabulate_to_latex(
    tabulate(
        dt.values, headers=[x for x in dt.columns], tablefmt="latex"
    ),
    "figures/rf_stats_table_0_.tex",
    "Random Forest tuning results.",
    2,
)

dt = pd.read_csv("data/rmse_rf.csv").round(2)
utils.tabulate_to_latex(
    tabulate(
        dt.values, headers=[x for x in dt.columns], tablefmt="latex"
    ),
    "figures/rf_stats_table_1_.tex",
    "Random Forest fitting results. Values shown are root mean square error.",
    0,
)
