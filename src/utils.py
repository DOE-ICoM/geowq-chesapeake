import numpy as np


def select_var(dt, var_col):
    # drop columns associated with all other vars from dt
    dt = dt.copy()
    var_key = ["SST (C)", "depth (m)", "SSS (psu)", "turbidity (NTU)"]
    var_key.remove(var_col)
    dt = dt.loc[:, ~dt.columns.str.startswith(tuple(var_key))]
    dt.replace([np.inf, -np.inf], np.nan, inplace=True)
    return dt


def freq_count(dt, grp_cols):
    dt = dt.copy()
    dt = dt.groupby(grp_cols).size().reset_index().rename(columns={
        0: 'count'
    }).sort_values("count", ascending=False).reset_index(drop=True)
    return dt
