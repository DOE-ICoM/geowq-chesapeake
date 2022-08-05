# https://gis.stackexchange.com/a/78699/32531

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from skimage.graph import route_through_array


def get_idx_coords(lon, lat):
    # lon = stations.iloc[[0]]["longitude"][0]
    # lat = stations.iloc[[0]]["latitude"][0]
    dt_where = dt.sel(x=lon, y=lat, method='nearest')
    dt_where = dt.where(
        (dt.x == float(dt_where["x"])) & (dt.y == float(dt_where["y"])),
        drop=True).squeeze()
    return (int(dt_where["x_idx"]), int(dt_where["y_idx"]))


def get_idx_water():
    dt_where = dt.where(
        dt.band_data == 1.0).to_array().values[0, :, :]  # , drop=True
    dt_where_water = np.argwhere(~np.isnan(dt_where))
    return dt_where_water


def get_distance(stop_idx_y, stop_idx_x):
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
    gdf = gpd.GeoDataFrame(df.cost, geometry=gpd.points_from_xy(df.x, df.y))

    return gdf


stations = gpd.GeoDataFrame({
    "name": ['Choptank', 'Susquehanna', 'Pautexent', 'Potomac'],
    "longitude": [-75.900, -76.083, -76.698, -77.059],
    "latitude": [38.811, 39.553, 38.8311, 39.379]
})
stations = gpd.GeoDataFrame(stations,
                            geometry=gpd.points_from_xy(
                                stations["longitude"], stations["latitude"]))

fpath = "data/cost.tif"
dt = xr.open_dataset(fpath, engine="rasterio").sel(band=1)
dt = dt.assign_coords(x_idx=dt["x"] * 0 + [x for x in range(0, 1038)],
                      y_idx=dt["y"] * 0 + [x for x in range(0, 1448)])
cost_surface_array = dt.to_array().values[0, :, :]

# for start_pnt i
(start_idx_x, start_idx_y) = get_idx_coords(stations.iloc[[0]]["longitude"][0],
                                            stations.iloc[[0]]["latitude"][0])
end_points = get_idx_water()

res = []
for i in tqdm(range(0, end_points.shape[0])):
    res.append(get_distance(end_points[i][0], end_points[i][1]))

# sum each layer
# res.rio.to_raster("data/testasdf35222.tif")
# gdf.to_file("test2.gpkg", driver="GPKG")
