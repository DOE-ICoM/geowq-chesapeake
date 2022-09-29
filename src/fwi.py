import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import skimage
from skimage.graph import route_through_array


def get_idx_coords(lon, lat, dt):
    # lon = stations.iloc[[0]]["longitude"][0]
    # lat = stations.iloc[[0]]["latitude"][0]
    dt_where = dt.sel(x=lon, y=lat, method="nearest")
    dt_where = dt.where(
        (dt.x == float(dt_where["x"])) & (dt.y == float(dt_where["y"])),
        drop=True).squeeze()
    return (int(dt_where["x_idx"]), int(dt_where["y_idx"]))


def get_geo_coords(x, y, dt):
    # y = 1199
    # x = 310
    dt_where = dt.where((dt.x_idx == x) & (dt.y_idx == y), drop=True)
    return (float(dt_where["x"]), float(dt_where["y"]))


def get_distance(stop_idx_y,
                 stop_idx_x,
                 i,
                 start_idx_y,
                 start_idx_x,
                 cost_surface_array,
                 dt,
                 return_full_path=False):
    try:
        indices, weight = route_through_array(
            cost_surface_array,
            (start_idx_y, start_idx_x),
            (stop_idx_x, stop_idx_y),
            geometric=True,
            fully_connected=True,
        )
        indices = np.array(indices).T
        weight = int(round(weight, 0))

        path = np.zeros_like(cost_surface_array)
        # set ONLY end point to weight
        path[np.array([indices[0][-1]]), np.array([indices[1][-1]])] = weight
        # path[indices[0], indices[1]] = weight

        res = xr.DataArray(path,
                           coords=[dt.y.values, dt.x.values],
                           dims=["y", "x"])
        res["spatial_ref"] = dt["spatial_ref"]
        res = res.where(res > 0)

        df = res.to_dataframe("cost").reset_index()
        df = df[~pd.isna(df["cost"])]
        gdf = gpd.GeoDataFrame({
            "cost": df.cost,
            "i": i
        },
                               geometry=gpd.points_from_xy(df.x, df.y))

        if return_full_path:
            full_path = path
            for i in range(0, indices.shape[1]):
                full_path[np.array([indices[0][i]]),
                          np.array([indices[1][i]])] = range(0, indices.shape[1])[i]            

            res_full = xr.DataArray(full_path,
                                    coords=[dt.y.values, dt.x.values],
                                    dims=["y", "x"])
            res_full["spatial_ref"] = dt["spatial_ref"]
            res_full = res_full.where(res_full > 0)
            df = res_full.to_dataframe("cost").reset_index()
            df = df[~pd.isna(df["cost"])]
            gdf = gpd.GeoDataFrame({
                "cost": df.cost,
                "i": i
            },
                                   geometry=gpd.points_from_xy(df.x, df.y))

    except:
        (lon, lat) = get_geo_coords(stop_idx_x, stop_idx_y, dt)
        gdf = gpd.GeoDataFrame({
            "cost": 999,
            "i": i
        },
                               geometry=gpd.points_from_xy(lon, lat))

    return gdf