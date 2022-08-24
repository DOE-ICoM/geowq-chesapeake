import os
import sys
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
    dt_sub = dt_sub[~pd.isna(dt_sub[variable])]

    res = dt_sub[dt_sub["variable"] == variable].groupby(
        ["loc_id", "latitude", "longitude"]
    ).size().reset_index().rename(columns={0: "count"}).sort_values(
        "count", ascending=False
    ).reset_index(
        drop=True
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
all(dt_grps[0]["count"] == dt_grps[1]["count"])

dt_grps[2].head()
dt_grps[1].head()



fig, axs = plt.subplots(ncols=3, nrows=1, subplot_kw={"projection": ccrs.PlateCarree()})

for i in range(0, len(axs)):
    ax = axs[i]
    gdf = dt_grps[i]
    if i == len(axs) - 1:
        gdf.h3.geo_to_h3_aggregate(6).plot("count", ax=ax, legend=True)
    else:
        gdf.h3.geo_to_h3_aggregate(6).plot("count", ax=ax, legend=False)
    ax.coastlines(resolution="10m", color="black", linewidth=1)

plt.show()
plt.savefig("figures/freqcount_hex.pdf")
