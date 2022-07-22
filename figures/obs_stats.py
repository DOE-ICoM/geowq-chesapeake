import os
import h3pandas
import pandas as pd
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from src import utils

variables = ["SSS (psu)", "turbidity (NTU)", "SST (C)"]

# --- raw data histograms by variable
dt_raw = pd.read_csv(
    os.environ["ICOM_DATA"] +
    "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv")

dt_melt = pd.melt(dt_raw[variables + ["pixelday"]],
                  id_vars=["pixelday"],
                  value_vars=variables)
[  # check dups
    any(dt_melt[dt_melt["variable"] == variables[i]]["pixelday"].duplicated())
    for i in range(0, len(variables))
]
g = sns.FacetGrid(dt_melt, col="variable", sharex=False)
g.map(sns.histplot, "value")
plt.savefig("figures/obs_varhist.pdf")

# --- obs counts table
dt_filtered = pd.read_csv(
    os.environ["ICOM_DATA"] +
    "/Modeling Data/Processed Data p1/filtered_w_data.csv",
    parse_dates=['datetime'])
dt_filtered["daystamp"] = [
    x.strftime("%Y-%m-%d") for x in dt_filtered.datetime
]


def _count_stats(dt, variable):
    dt_sub = utils.select_var(dt, variable)
    dt_sub = dt_sub[~pd.isna(dt_sub[variable])]
    dt_sub = utils.freq_count(
        dt_sub, ['daystamp', 'source', 'station_id'])  # collapse to daily
    n_measures = utils.freq_count(dt_sub,
                                  ['daystamp', 'source', 'station_id'
                                   ]).sort_values(["station_id", "daystamp"],
                                                  ascending=False).shape[0]
    n_programs = len(dt_sub["source"].unique())
    n_stations = len(dt_sub["station_id"].unique())
    return pd.DataFrame(
        {
            "variable": variable,
            "n_measures": n_measures,
            "n_programs": n_programs,
            "n_stations": n_stations
        },
        index=[0])


obs_counts = pd.concat([_count_stats(dt_filtered, v)
                        for v in variables]).sort_values(["n_measures"],
                                                         ascending=False)
# obs_counts.to_clipboard()

# --- obs distribution stats

dt_agg = pd.read_csv(os.environ["ICOM_DATA"] +
                     "/Modeling Data/Processed Data p1/aggregated.csv",
                     parse_dates=['datetime'])
obs_distribution = pd.concat([
    pd.DataFrame(
        utils.select_var(dt_agg, v)[v].quantile([0, 0.05, 0.5, 0.95, 1])).T
    for v in variables
]).round(2)
# obs_distribution.to_clipboard()

# --- loc_id frequency count histograms and hex map
# dt_grp = dt_raw.groupby(['loc_id', "latitude",
#                          "longitude"]).size().reset_index().rename(columns={
#                              0: 'count'
#                          }).sort_values("count",
#                                         ascending=False).reset_index(drop=True)
# g = sns.histplot(dt_grp, x="count")
# dt_grp_gdf = gpd.GeoDataFrame(dt_grp,
#                               geometry=gpd.points_from_xy(
#                                   dt_grp["longitude"], dt_grp["latitude"]))

# plt.figure(figsize=(5, 5))
# ax = plt.axes(projection=ccrs.PlateCarree())
# dt_grp_gdf.h3.geo_to_h3_aggregate(6).plot("count", ax=ax)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# plt.show()
