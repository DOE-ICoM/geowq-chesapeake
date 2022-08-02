import sys
import pandas as pd

sys.path.append(".")
from src import utils

variables = ["SSS (psu)", "turbidity (NTU)", "SST (C)"]
variables_str = ["salinity (psu)", "turbidity (NTU)", "temperature (C)"]
variables_str_short = [utils.clean_var_name(v) for v in variables_str]

rf_rmse = pd.concat([
    pd.DataFrame(variables_str_short, columns=["variable"
                                               ]).reset_index(drop=True),
    pd.concat([utils.load_md(v, "data/rmse_")
               for v in variables_str_short]).reset_index(drop=True)
],
                    axis=1).T
rf_rmse = rf_rmse.iloc[[1]]
rf_rmse.columns = [variables_str_short]
rf_rmse.to_csv("data/rmse_rf.csv", index=False)
# rf_rmse.to_clipboard()