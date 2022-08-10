# https://gis.stackexchange.com/a/78699/32531

import os
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from joblib import Parallel, delayed
from skimage.graph import route_through_array


def get_idx_coords(lon, lat):
    # lon = stations.iloc[[0]]["longitude"][0]
    # lat = stations.iloc[[0]]["latitude"][0]
    dt_where = dt.sel(x=lon, y=lat, method='nearest')
    dt_where = dt.where(
        (dt.x == float(dt_where["x"])) & (dt.y == float(dt_where["y"])),
        drop=True).squeeze()
    return (int(dt_where["x_idx"]), int(dt_where["y_idx"]))

def get_geo_coords(x, y):
    dt_where = dt.where((dt.x_idx==x) & (dt.y_idx==y), drop=True)
    return (float(dt_where["x"]), float(dt_where["y"]))

def get_idx_water():
    dt_where = dt.where(dt.band_data == 1.0)
    # dt_where.rio.to_raster("test2.tif")
    dt_where = dt_where.to_array().values[0, :, :]  # , drop=True
    dt_where_water = np.argwhere(~np.isnan(dt_where))
    return dt_where_water


def get_distance(stop_idx_y, stop_idx_x, i):
    try:
        indices, weight = route_through_array(cost_surface_array,
                                            (start_idx_y, start_idx_x),
                                            (stop_idx_y, stop_idx_x),
                                            geometric=True,
                                            fully_connected=True)
        indices = np.array(indices).T
        weight = int(round(weight, 0))

        path = np.zeros_like(cost_surface_array)
        path[np.array([indices[0][-1]]), np.array([indices[1][-1]])] = weight
        # path[indices[0], indices[1]] = weight

        res = xr.DataArray(path,
                        coords=[dt.y.values, dt.x.values],
                        dims=["y", "x"])
        res["spatial_ref"] = dt["spatial_ref"]
        res = res.where(res > 0)

        df = res.to_dataframe("cost").reset_index()
        df = df[~pd.isna(df["cost"])]
        gdf = gpd.GeoDataFrame({"cost": df.cost, "i":i}, geometry=gpd.points_from_xy(df.x, df.y))

    except:
        gdf = gpd.GeoDataFrame({"cost": 999, "i":i}, index=[0])

    return gdf
    


def get_distance_grp(rng):
    last_rng = rng[-1]
    outpath = "test_" + str(last_rng) + ".gpkg"
    if not os.path.exists(outpath):
        par_njobs = 10  # , verbose=25
        gdfs = Parallel(n_jobs=par_njobs)(delayed(get_distance)(end_points[i][0],
                                                    end_points[i][1], i)
                            for i in rng)  # end_points.shape[0]

        pd.concat(gdfs).to_file(outpath, driver="GPKG")


stations = gpd.GeoDataFrame({
    "name": ['Choptank', 'Susquehanna', 'Pautexent', 'Potomac'],
    "longitude": [-75.900, -76.083, -76.692, -77.110],
    "latitude": [38.811, 39.553, 38.664, 38.38]
})
stations = gpd.GeoDataFrame(stations,
                            geometry=gpd.points_from_xy(
                                stations["longitude"], stations["latitude"]))
# stations.to_file("stations.gpkg", driver="GPKG")

fpath = "data/cost.tif"
dt = xr.open_dataset(fpath, engine="rasterio").sel(band=1)
dt = dt.assign_coords(x_idx=dt["x"] * 0 + [x for x in range(0, 1038)],
                      y_idx=dt["y"] * 0 + [x for x in range(0, 1448)])
cost_surface_array = dt.to_array().values[0, :, :]

# for start_pnt i
(start_idx_x, start_idx_y) = get_idx_coords(stations.iloc[[0]]["longitude"][0],
                                            stations.iloc[[0]]["latitude"][0])
end_points = get_idx_water()

# ----


end_points.shape[0]  # 294049
l = np.array_split(np.array(range(0, end_points.shape[0])), 6)
[s[-1] for s in l]
# [get_distance_grp(s) for s in l]

res = []
for i in tqdm(l[-1]):
    # print(i)
    res.append(get_distance(end_points[i][0], end_points[i][1], i))

# i = 11311
# get_distance(end_points[i][0], end_points[i][1], i)
# (end_points[11311][0], end_points[11311][1])
# get_geo_coords(end_points[11311][0], end_points[11311][1])

# sum each layer
# res.rio.to_raster("data/testasdf35222.tif")
# gdf.to_file("test2.gpkg", driver="GPKG")
