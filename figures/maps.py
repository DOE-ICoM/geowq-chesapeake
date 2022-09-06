import os
import sys
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
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

    res = (dt_sub[dt_sub["variable"] == variable].groupby(
        ["loc_id", "latitude",
         "longitude"]).size().reset_index().rename(columns={
             0: "count"
         }).sort_values("count", ascending=False).reset_index(drop=True))

    res_gdf = gpd.GeoDataFrame(res,
                               geometry=gpd.points_from_xy(
                                   res["longitude"], res["latitude"]))

    return res_gdf


dt_raw = pd.read_csv(
    os.environ["ICOM_DATA"] +
    "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv")
id_vars = ["loc_id", "latitude", "longitude"]
variables = ["SSS (psu)", "turbidity (NTU)", "SST (C)"]
dt_filtered = dt_raw[id_vars + variables]
dt_melt = pd.melt(dt_filtered, id_vars=id_vars, value_vars=variables)

dt_grps = [get_pnt_counts(dt_melt, variable) for variable in variables]

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

# --- input/output map
bay_gdf_hires = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(
    epsg=4326)

# get cbofs image
tod = "20220904"
tif_path = "data/cbofs/salt_{date}.tif".format(date=tod)
img_cbofs = xr.open_dataset(tif_path)
img_cbofs = img_cbofs.rio.clip(bay_gdf_hires.geometry)
img_cbofs = img_cbofs["band_data"].sel(band=1)
# img_cbofs.plot.imshow()
# plt.show()

# get rf prediction image
# gdalwarp -te -77.3425000000000011 36.1675000000000040 -74.7974999999999994 39.6325000000000074 -ts 509 693 -overwrite 2022-09-04.tif 2022-09-04_downsample.tif

date = "2022-09-04"
img_rf = xr.open_dataset("data/prediction/" + date + "_downsample.tif",
                         engine="rasterio")
img_rf = img_rf.rio.clip(bay_gdf_hires.geometry)
img_rf = img_rf["band_data"].sel(band=1)
img_rf.rio.to_raster("data/prediction/" + date + "_downsample_clip.tif")
# img_rf.plot.imshow()
# plt.show()

# gdal_calc.py -a data/prediction/2022-09-04_downsample_clip.tif -b data/cbofs/salt_20220904.tif --calc="a - b" --outfile c.tif

# get GEE image of sur_refl_08 band
date = "2022-09-03"
img_gee = xr.open_dataset(utils.modisaqua_path(date), engine="rasterio")
img_gee = img_gee.rio.clip(bay_gdf_hires.geometry)
img_gee = img_gee["band_data"].sel(band=1)
# img_gee.plot.imshow()
# plt.show()

# create raster diff
lons = img_rf.sortby("y", "x").x.as_numpy().values
lats = img_rf.sortby("y", "x").y.as_numpy().values
data = img_rf.sortby("y", "x").to_numpy() - img_cbofs.sortby("y",
                                                             "x").to_numpy()
test = xr.DataArray(data=data,
                    dims=["y", "x"],
                    coords=dict(x=(lons), y=(lats)))
test = test.rio.set_spatial_dims("x", "y")


def panel_add(i, axs, title, geo_grid, diff=False):
    ax = axs[i]
    if diff:
        geo_grid.plot.imshow(ax=ax,
                             center=0,
                             cbar_kwargs={
                                 "shrink": 0.5,
                                 'label': ''
                             })
    else:
        geo_grid.plot.imshow(ax=ax,
                             vmin=0,
                             cbar_kwargs={
                                 "shrink": 0.5,
                                 'label': ''
                             })  # , vmax=np.nanmax(img_cbofs.to_numpy()))
    ax.set_title(title)
    ax.coastlines(resolution="10m", color="black", linewidth=1)


fig, axs = plt.subplots(
    ncols=3,
    nrows=1,
    constrained_layout=True,
    subplot_kw={
        "projection": ccrs.PlateCarree(),
        "xlabel": ""
    },
)

# st = fig.suptitle("Number of days for each month meeting the criteria in 2017", fontsize="large")
panel_add(0, axs, "RF Results", img_rf)
panel_add(1, axs, "CBOFS Snapshot", img_cbofs)
panel_add(2, axs, "RF-CBOFS", test, diff=True)

# plt.show()
plt.savefig("figures/_rf-vs-cbofs.pdf")
