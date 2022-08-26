import re
import numpy as np
import pandas as pd


def select_var(dt, var_col):
    if isinstance(dt, str):
        dt = pd.read_csv(dt)
    dt = dt.copy()

    # drop columns associated with all other vars from dt
    var_key = ["SST (C)", "depth (m)", "SSS (psu)", "turbidity (NTU)"]
    var_key.remove(var_col)

    dt = dt.loc[:, ~dt.columns.str.startswith(tuple(var_key))]
    dt.replace([np.inf, -np.inf], np.nan, inplace=True)

    return dt


def freq_count(dt, grp_cols):
    dt = dt.copy()
    dt = (
        dt.groupby(grp_cols)
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return dt


def _unwrap(y):
    return [[x for x in re.split("\s+", y)]]


def load_md(variable, fpath_prestring="data/best_params_"):
    # load best_params_[variable].md
    bp_colnames = _unwrap(
        str(
            pd.read_table(fpath_prestring + variable + ".md", nrows=1, header=None)[0][
                0
            ]
        ).strip()
    )[0]
    if "max_leaf_nodes" in bp_colnames:
        bp_colnames.remove("max_leaf_nodes")
    bp = pd.DataFrame(
        _unwrap(
            str(
                pd.read_table(
                    fpath_prestring + variable + ".md", skiprows=2, header=None
                )[0][0]
            ).strip()
        ),
        columns=bp_colnames,
    )
    return bp


def clean_var_name(x):
    return "".join(map(str.lower, x)).split(" ")[0]


def datetime_to_doy(dt):
    dt = dt.copy()
    dt = pd.to_datetime(dt)
    ## Subtract from fitted sine wave/ Calculate day of Year instead
    #    data['datetime']=pd.to_datetime(data['datetime'], infer_datetime_format=True)
    dt = dt.astype("datetime64")
    dt = dt.dt.strftime("%j")
    dt = pd.to_numeric(dt, downcast="float")
    return dt
