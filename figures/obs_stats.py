import os
import sys
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt

sys.path.append(".")
from src import utils

variables = ["SSS (psu)", "SST (C)", "turbidity (NTU)"]

# --- raw data histograms by variable
dt_raw = pd.read_csv(
    os.environ["ICOM_DATA"]
    + "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv"
)

# remove turbidity below 0
dt_raw.loc[dt_raw["turbidity (NTU)"] < 0, "turbidity (NTU)"] = None

# replace salinity less than 0 with 0
dt_raw.loc[dt_raw["SSS (psu)"] < 0, "SSS (psu)"] = 0


dt_melt = pd.melt(
    dt_raw[variables + ["pixelday"]], id_vars=["pixelday"], value_vars=variables
)
[  # check dups
    any(dt_melt[dt_melt["variable"] == variables[i]]["pixelday"].duplicated())
    for i in range(0, len(variables))
]
g = sns.FacetGrid(dt_melt, col="variable", sharex=False)
g.map(sns.histplot, "value")
g.axes[0][1].set_xscale("log")
# plt.show()
plt.savefig("figures/obs_varhist.pdf")

# --- obs counts table
dt_filtered = pd.read_csv(
    os.environ["ICOM_DATA"] + "/Modeling Data/Processed Data p1/filtered_w_data.csv",
    parse_dates=["datetime"],
)
dt_filtered["daystamp"] = [x.strftime("%Y-%m-%d") for x in dt_filtered.datetime]


def _count_stats(dt, variable):
    dt_sub = utils.select_var(dt, variable)
    dt_sub = dt_sub[~pd.isna(dt_sub[variable])]
    dt_sub = utils.freq_count(
        dt_sub, ["daystamp", "source", "station_id"]
    )  # collapse to daily

    # What are the top 3 programs and what percentage of obs do they account for?
    utils.freq_count(dt_sub, ["source"]).head(3)
    sum(utils.freq_count(dt_sub, ["source"]).iloc[0:3]["count"]) / sum(utils.freq_count(dt_sub, ["source"])["count"])

    n_measures = (
        utils.freq_count(dt_sub, ["daystamp", "source", "station_id"])
        .sort_values(["station_id", "daystamp"], ascending=False)
        .shape[0]
    )
    n_programs = len(dt_sub["source"].unique())
    n_stations = len(dt_sub["station_id"].unique())
    return pd.DataFrame(
        {
            "variable": variable,
            "n_measures": n_measures,
            "n_programs": n_programs,
            "n_stations": n_stations,
        },
        index=[0],
    )


obs_counts = pd.concat([_count_stats(dt_filtered, v) for v in variables]).sort_values(
    ["n_measures"], ascending=False
)
print(obs_counts)
with open("figures/obs_counts.md", "w") as f:
    f.write(tabulate(obs_counts.values, headers=[x for x in obs_counts.columns]))
utils.tabulate_to_latex(
    tabulate(
        obs_counts.values, headers=[x for x in obs_counts.columns], tablefmt="latex"
    ),
    "figures/counts_obs.tex",
    "Number of measurements, programs, and stations in the observational dataset.",
    0,
)

# obs_counts.to_clipboard()

# --- obs distribution stats
dt_agg = pd.read_csv(
    os.environ["ICOM_DATA"] + "/Modeling Data/Processed Data p1/aggregated.csv",
    parse_dates=["datetime"],
)

# per call_data2.py,
# for turbidity get rid of negative numbers
dt_agg.loc[dt_agg["turbidity (NTU)"] < 0, "turbidity (NTU)"] = None
# for salinity set negative numbers to 0
dt_agg.loc[dt_agg["SSS (psu)"] < 0, "SSS (psu)"] = 0

obs_distribution = (
    pd.concat(
        [
            pd.DataFrame(
                utils.select_var(dt_agg, v)[v].quantile([0, 0.05, 0.5, 0.95, 1])
            ).T
            for v in variables
        ]
    )
    .round(2)
    .reset_index()
    .rename(columns={"index": "variable"})
)
print(obs_distribution)
with open("figures/obs_distribution.md", "w") as f:
    f.write(
        tabulate(obs_distribution.values, headers=[x for x in obs_distribution.columns])
    )
# obs_distribution.to_clipboard()
utils.tabulate_to_latex(
    tabulate(
        obs_distribution.values, headers=[x for x in obs_distribution.columns], tablefmt="latex"
    ),
    "figures/obs_distribution.tex",
    "Distribution of values in the observational dataset.",
    1,
)