import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import h3pandas  # h3.geo_to_h3_aggregate

sys.path.append(".")
from src import utils

# --- loc_id frequency count histograms and hex map
def get_pnt_counts(dt, variable):
    # variable = variables[0]
    dt_sub = utils.select_var(dt, variable)
    dt_sub = dt_sub[~pd.isna(dt_sub["value"])]

    res = (
        dt_sub[dt_sub["variable"] == variable]
        .groupby(["loc_id", "latitude", "longitude"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    res_gdf = gpd.GeoDataFrame(
        res, geometry=gpd.points_from_xy(res["longitude"], res["latitude"])
    )

    return res_gdf


dt_raw = pd.read_csv(
    os.environ["ICOM_DATA"]
    + "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv"
)
id_vars = ["loc_id", "latitude", "longitude"]
variables = ["SSS (psu)", "turbidity (NTU)", "SST (C)"]
dt_filtered = dt_raw[id_vars + variables]
dt_melt = pd.melt(dt_filtered, id_vars=id_vars, value_vars=variables)

dt_grps = [get_pnt_counts(dt_melt, variable) for variable in variables]

# ---

fig, axs = plt.subplots(
    ncols=3,
    nrows=1,
    constrained_layout=True,
    subplot_kw={"projection": ccrs.PlateCarree()},
)

mins = []
maxs = []
gdf_aggs = []
for i in range(0, len(dt_grps)):
    gdf = dt_grps[i]
    gdf_agg = gdf.h3.geo_to_h3_aggregate(6)
    mins.append(min(gdf_agg.reset_index()["count"]))
    maxs.append(max(gdf_agg.reset_index()["count"]))
    gdf_aggs.append(gdf_agg)

for i in range(0, len(axs)):
    ax = axs[i]
    ax.set_title(variables[i])
    gdf_agg = gdf_aggs[i]
    gdf_agg.plot("count", ax=ax, legend=False, vmax=max(maxs))
    ax.coastlines(resolution="10m", color="black", linewidth=1)

scales = np.linspace(1, max(maxs), 7)
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(scales.min(), scales.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs[len(axs) - 1], shrink=0.78)
cbar.ax.set_title("obs count (n)", y=-0.08)

# plt.show()
plt.savefig("figures/_freqcount_hex.pdf")
