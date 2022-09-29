import sys
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.geometry import LineString

sys.path.append("src")
from src import fwi

stations = gpd.read_file("stations.gpkg", driver="GPKG")
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
test["weight"] = np.log(test["discharge_va"] * 4)

a = test["weight"]
alphas = np.interp(a, (a.min(), a.max()), (0.7, +1))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent, ccrs.PlateCarree())
ax.coastlines(resolution="10m", color="black", linewidth=1)
test.plot(ax=ax, column="tributary", markersize="weight", legend=True)
plt.show()

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution="10m", color="black", linewidth=1, alpha=1)
test[test["tributary"] == "Choptank"].plot(ax=ax,                                           
                                           markersize="weight",
                                           alpha=0.1,
                                           color="blue")
test[test["tributary"] == "Potomac"].plot(ax=ax,                                           
                                           markersize="weight",
                                           alpha=1,
                                           color="blue")
plt.show()