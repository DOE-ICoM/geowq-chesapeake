# https://gis.stackexchange.com/a/78699/32531

import numpy as np
import xarray as xr
import geopandas as gpd
from skimage.graph import route_through_array


def get_idx(lon, lat):
    # lon = stations.iloc[[0]]["longitude"][0]
    # lat = stations.iloc[[0]]["latitude"][0]
    dt_where = dt.sel(x=lon, y=lat, method='nearest')
    dt_where = dt.where(
        (dt.x == float(dt_where["x"])) & (dt.y == float(dt_where["y"])),
        drop=True).squeeze()
    return (int(dt_where["x_idx"]), int(dt_where["y_idx"]))


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

(start_idx_x, start_idx_y) = get_idx(stations.iloc[[0]]["longitude"][0],
                                     stations.iloc[[0]]["latitude"][0])
# dt.isel(x=783, y=399)
(stop_idx_x,
 stop_idx_y) = get_idx(stations.iloc[[1]].reset_index()["longitude"][0],
                       stations.iloc[[1]].reset_index()["latitude"][0])

cost_surface_array = dt.to_array().values[0, :, :]
indices, weight = route_through_array(cost_surface_array,
                                      (start_idx_y, start_idx_x),
                                      (stop_idx_y, stop_idx_x),
                                      geometric=True,
                                      fully_connected=True)
indices = np.array(indices).T

path = np.zeros_like(cost_surface_array)
path[np.array([indices[0][0]]), np.array([indices[1][0]])] = int(round(weight, 0))
# path[indices[0], indices[1]] = weight

res = xr.DataArray(path, coords=[dt.y.values, dt.x.values], dims=["y", "x"])
res["spatial_ref"] = dt["spatial_ref"]
res = res.where(res > 0)

# res.rio.to_raster("data/testasdf35.tif")