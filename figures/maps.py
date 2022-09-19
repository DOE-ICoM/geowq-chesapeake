import os
import sys
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import gridspec
import h3pandas  # h3.geo_to_h3_aggregate

sys.path.append(".")
from src import utils


def panel_add(i,
              axs,
              title,
              geo_grid,
              diff=False,
              j=None,
              vmax=27,
              vmin=0,
              ticks=None,
              height_frac=0.5):
    if j is not None:
        ax = plt.subplot(axs[i, j], xlabel="", projection=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="black", linewidth=1)
    else:
        ax = axs[i]
        ax.coastlines(resolution="10m", color="black", linewidth=1)
    if diff:
        geo_grid.plot.imshow(ax=ax,
                             center=0,
                             cbar_kwargs={
                                 "shrink": 0.5,
                                 'label': ''
                             })
    else:
        geo_grid.plot.imshow(ax=ax,
                             vmin=vmin,
                             vmax=vmax,
                             cbar_kwargs={
                                 "shrink": 0.5,
                                 'label': '',
                                 'ticks': ticks,
                             })  # , vmax=np.nanmax(img_cbofs.to_numpy()))
    ax.set_title(title,
                 size="small",
                 y=height_frac,
                 x=0.9,
                 rotation="vertical")


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

# --- rf vs cbofs map
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

img_rf = utils.get_rf_prediction("2022-09-04", "salinity")

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
panel_add(0, axs, "RF Results", img_rf, height_frac=0.65)
panel_add(1, axs, "CBOFS Snapshot", img_cbofs)
panel_add(2, axs, "RF-CBOFS", test, diff=True, height_frac=0.7)

# plt.show()
plt.savefig("figures/_rf-vs-cbofs.pdf")

# --- seasonality maps
def _seasonality_map(variable, vmin=(0,0,0,0), vmax=(27,27,27,27), ticks=None):
    # variable  = "turbidity"
    # vmin=(3,3,21,21)
    # vmax=(12, 12, 29, 29)
    dates = ["2021-12-04", "2022-03-04", "2022-06-04", "2022-09-04"]
    imgs = [utils.get_rf_prediction(date, variable) for date in dates]

    nrow = 2
    ncol = 2

    fig = plt.figure(figsize=(ncol + 3, nrow + 3))
    axs = gridspec.GridSpec(nrow,
                            ncol,
                            wspace=0.0,
                            hspace=0.0,
                            top=1. - 0.5 / (nrow + 1),
                            bottom=0.5 / (nrow + 1),
                            left=0.1666,
                            right=0.7)  # 0.83333

    panel_add(0, axs, dates[0], imgs[0], j=0, vmin=vmin[0], vmax=vmax[0], ticks=ticks)
    panel_add(0, axs, dates[1], imgs[1], j=1, vmin=vmin[1], vmax=vmax[1], ticks=ticks)
    panel_add(1, axs, dates[2], imgs[2], j=0, vmin=vmin[2], vmax=vmax[2], ticks=ticks)
    panel_add(1, axs, dates[3], imgs[3], j=1, vmin=vmin[3], vmax=vmax[3], ticks=ticks)
    # rm first column colorbars
    # fig.axes
    # fig.delaxes(fig.axes[1])
    # fig.delaxes(fig.axes[4])

    # fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    out_path = "figures/_seasonality_" + variable + ".pdf"
    fig.suptitle(variable, y=0.87, x=0.42)
    plt.savefig(out_path)

    return out_path

_seasonality_map("salinity", ticks=[0,10,20])
_seasonality_map("temperature", vmin=(3,3,21,21), vmax=(12, 12, 29, 29))
_seasonality_map("turbidity", vmin=(0,0,0,0), vmax=(40, 40, 40, 40))
