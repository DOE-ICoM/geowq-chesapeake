import sys
import pickle
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib import gridspec
import matplotlib.pyplot as plt
from xrspatial.classify import quantile
from shapely.geometry import LineString

sys.path.append("src")
from src import fwi

# --- fwi "flow" diagram

fpath = "data/cost.tif"
dt = xr.open_dataset(fpath, engine="rasterio").sel(band=1)
dt = dt.assign_coords(
    x_idx=dt["x"] * 0 + [x for x in range(0, 1038)],
    y_idx=dt["y"] * 0 + [x for x in range(0, 1448)],
)
cost_surface_array = dt.to_array().values[0, :, :]


def _get_path(j, i=171133):
    # j = 1
    (start_idx_x, start_idx_y) = fwi.get_idx_coords(
        stations.iloc[[j]]["longitude"].reset_index(drop=True)[0],
        stations.iloc[[j]]["latitude"].reset_index(drop=True)[0], dt)

    end_points = pickle.load(open("data/end_points.pkl", "rb"))

    path = fwi.get_distance(end_points[i][1],
                            end_points[i][0],
                            i,
                            start_idx_y,
                            start_idx_x,
                            cost_surface_array,
                            dt,
                            return_full_path=True)
    path["tributary"] = stations.iloc[[j]]["name"].reset_index(drop=True)[0]
    return path


stations = gpd.read_file("stations.gpkg", driver="GPKG")
res = {
    stations.iloc[[j]]["name"].reset_index(drop=True)[0]: _get_path(j)
    for j in range(0, stations.shape[0])
}

b = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(epsg=4326)
b_bounds = [x for x in b.bounds.iloc[0]]
extent = (b_bounds[0], b_bounds[2], b_bounds[1], b_bounds[3])

test_lines = pd.concat([
    gpd.GeoDataFrame(
        {"tributary": key},
        geometry=[LineString(res[key]['geometry'].reset_index(drop=True))],
        index=[0]) for key in res.keys()
])
test = pd.concat([res[key] for key in res.keys()])
discharge = pd.read_csv("data/discharge_median.csv")
test = test.merge(discharge, left_on="tributary", right_on="site_str")

max_cost = test.groupby("tributary").max("cost").reset_index().rename(
    columns={"cost": "max_cost"})[["tributary", "max_cost"]]
test = test.merge(max_cost)

test["weight"] = test["discharge_va"] * test["max_cost"]
test["weight"] = np.interp(test["weight"],
                           (test["weight"].min(), test["weight"].max()),
                           (0.1, +8))
np.unique(test["weight"])

# # simple non-arrowed
# fig = plt.figure(figsize=[3.4 / 1.4, 2.8 / 1.4])
# ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
# ax.set_extent(extent, ccrs.PlateCarree())
# ax.coastlines(resolution="10m", color="black", linewidth=1)
# test.plot(ax=ax, column="tributary", markersize="weight", alpha=0.6)
# plt.savefig("figures/_fwi.pdf", bbox_inches='tight')

# ---

test_split = [
    test[test["tributary"] == trib] for trib in np.unique(test["tributary"])
]
ends = pd.concat([x[x["cost"] == np.max(x["cost"])] for x in test_split])
ends_minus_one = [{
    "cost": ends.iloc[[i]]["max_cost"].values[0] - 1,
    "tributary": ends.iloc[[i]]["tributary"].values[0]
} for i in range(0, ends.shape[0])]
ends_minus_one = pd.concat([
    test[(test["tributary"] == x["tributary"]) & (test["cost"] == x["cost"])]
    for x in ends_minus_one
])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[i] for i in [0, 2, 5, 7, 9]]

# # fancier arrowed version
fig = plt.figure(figsize=[3.4 / 1.4, 2.8 / 1.4])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(extent, ccrs.PlateCarree())
ax.coastlines(resolution="10m", color="black", linewidth=1)
test.plot(ax=ax, column="tributary", markersize="weight", alpha=0.6)
# https://matplotlib.org/2.0.2/users/annotations.html
[
    ax.annotate(text="",
                xy=ends.iloc[i].geometry.centroid.coords[0],
                xytext=ends_minus_one.iloc[i].geometry.centroid.coords[0],
                size=10,
                alpha=0.6,
                arrowprops=dict(facecolor=colors[i],
                                arrowstyle="simple",
                                edgecolor=colors[i],
                                connectionstyle="arc3"))
    for i in range(0, ends.shape[0])
]
# plt.show()
plt.savefig("figures/_fwi.pdf", bbox_inches='tight')

# --- fwi raster panels


def _process(x):
    merged = xr.concat([x], dim="band")
    merged = merged.sum(dim="band", skipna=True)
    merged = merged.where(merged > 0)
    merged = merged.where(merged < 5000)
    merged = merged.bfill("time")
    return (quantile(merged["cost"], k=5), merged["cost"])


discharge = pd.read_csv("data/discharge_median.csv")
tribs = ["susquehanna", "choptank"]
flist = ["data/waterdistance/" + trib + ".gpkg" for trib in tribs]
grids = [_process(fwi.weight_grid(f, discharge)) for f in flist]


def panel_add(i,
              axs,
              title,
              geo_grid,
              j=None,
              vmax=27,
              vmin=0,
              ticks=None,
              height_frac=0.5,
              format=None):
    if j is not None:
        ax = axs[i, j]
        ax = plt.subplot(ax, xlabel="", projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="black", linewidth=1)
    else:
        ax = axs[i]
        ax = plt.subplot(ax, xlabel="", projection=ccrs.PlateCarree())
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="black", linewidth=1)

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    cmap = matplotlib.colors.ListedColormap([
        "royalblue", "cornflowerblue", "lightsteelblue", "powderblue",
        "aliceblue"
    ])

    im = geo_grid.plot.imshow(ax=ax,
                              vmin=vmin,
                              vmax=vmax,
                              cmap=cmap,
                              cbar_kwargs={
                                  "shrink": 0.7,
                                  'label': '',
                                  'ticks': ticks
                              })

    cbar = im.colorbar
    cbar.ax.set_yticklabels(format)

    ax.set_title(title,
                 size="small",
                 y=height_frac,
                 x=0.9,
                 rotation="vertical")

    return ax


def line_add(i, j, i_ends, axes):
    ax = axes[i, j]
    ax = plt.subplot(ax, xlabel="", projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.coastlines(resolution="10m", color="black", linewidth=1)
    test[test["tributary"] == tribs[i].title()].plot(
        ax=ax,
        markersize="weight",
        alpha=0.6,
        color=list(reversed(colors))[i_ends])

    # https://matplotlib.org/2.0.2/users/annotations.html
    ax.annotate(text="",
                xy=ends.iloc[i_ends].geometry.centroid.coords[0],
                xytext=ends_minus_one.iloc[i_ends].geometry.centroid.coords[0],
                alpha=0.6,
                arrowprops=dict(facecolor=list(reversed(colors))[i_ends],
                                edgecolor="white",
                                linewidth=4,
                                headwidth=20,
                                connectionstyle="arc3"))
    return ax


ncol = 2
nrow = 2
fig = plt.figure(figsize=(ncol + 3, nrow + 3))
axes = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1, right=0.9)

line_add(0, 0, 4, axes)
line_add(1, 0, 0, axes)
cbar_labels = [
    [
        round(x, 3)
        for x in np.nanquantile(grids[0][1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    ],
    [
        round(x, 3)
        for x in np.nanquantile(grids[1][1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    ]
]
panel_add(0, axes, "", grids[0][0], j=1, vmax=5, format=cbar_labels[0])
panel_add(1, axes, "", grids[1][0], j=1, vmax=5, format=cbar_labels[1])

# plt.subplots_adjust(right=0)

plt.show()

# ---

ncol = 1
nrow = 1
fig = plt.figure(figsize=(ncol + 3, nrow + 3))
axes = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1, right=0.9)
panel_add(0, axes, "", grids[0][0], vmax=5, format=cbar_labels[0])
plt.show()