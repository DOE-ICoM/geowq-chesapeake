# https://gis.stackexchange.com/a/78699/32531

import os
import sys
import glob
import random
import pickle
import rasterio
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from joblib import Parallel, delayed

sys.path.append("src")
from src import fwi

def get_idx_water():
    """
    get_idx_water()
    """
    dt_where = dt.where(dt.band_data == 1.0)  # , drop=True
    # dt_where.rio.to_raster("test2.tif")
    dt_where = dt_where.to_array().values[0, :, :]
    dt_where_water = np.argwhere(~np.isnan(dt_where))
    return dt_where_water


# rng = l[-1]
# [(s[0][0], s[0][1], s[0][2]) for s in zip([(end_points[i][1], end_points[i][0], i) for i in rng])]


def get_distance_grp(rng):
    last_rng = rng[-1]
    outpath = "test_" + str(last_rng) + ".gpkg"
    if not os.path.exists(outpath):
        par_njobs = 12
        gdfs = Parallel(n_jobs=par_njobs, verbose=25, batch_size=2)(
            delayed(fwi.get_distance)(s[0][0], s[0][1], s[0][2], start_idx_y, start_idx_x, cost_surface_array, dt)
            for s in zip([(end_points[i][1], end_points[i][0], i) for i in rng])
        )

        pd.concat(gdfs).to_file(outpath, driver="GPKG")


stations = gpd.GeoDataFrame(
    {
        "name": ["Choptank", "Susquehanna", "Pautexent", "Potomac", "James"],
        "longitude": [-75.900, -76.069, -76.692, -77.110, -77.2767],
        "latitude": [38.811, 39.525, 38.664, 38.38, 37.3238],
    }
)
stations = gpd.GeoDataFrame(
    stations, geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"])
)
stations.to_file("stations.gpkg", driver="GPKG")

fpath = "data/cost.tif"
dt = xr.open_dataset(fpath, engine="rasterio").sel(band=1)
dt = dt.assign_coords(
    x_idx=dt["x"] * 0 + [x for x in range(0, 1038)],
    y_idx=dt["y"] * 0 + [x for x in range(0, 1448)],
)
cost_surface_array = dt.to_array().values[0, :, :]

if not os.path.exists("data/end_points.pkl"):
    end_points = get_idx_water()
    pickle.dump(end_points, open("data/end_points.pkl", "wb"))
end_points = pickle.load(open("data/end_points.pkl", "rb"))

# --- james
if not os.path.exists("data/waterdistance/james.gpkg"):
    (start_idx_x, start_idx_y) = fwi.get_idx_coords(
        stations.iloc[[4]].reset_index()["longitude"][0],
        stations.iloc[[4]].reset_index()["latitude"][0],
        dt
    )
    l = np.array_split(np.array(range(0, end_points.shape[0])), 6)    
    [get_distance_grp(s) for s in l]

    flist = glob.glob("test_*.gpkg")
    gdfs = [gpd.read_file(f) for f in flist]
    pd.concat(gdfs).to_file("james.gpkg")
    [os.remove(f) for f in flist]


# --- choptank
if not os.path.exists("data/waterdistance/choptank.gpkg"):
    (start_idx_x, start_idx_y) = fwi.get_idx_coords(
        stations.iloc[[0]]["longitude"][0], stations.iloc[[0]]["latitude"][0], dt
    )
    # end_points.shape[0]  # 294049
    l = np.array_split(np.array(range(0, end_points.shape[0])), 6)
    l[-1] = np.delete(l[-1], np.array([11311]))  # resolve mystery coredump error
    # [s[-1] for s in l]
    [get_distance_grp(s) for s in l]

    flist = glob.glob("test_*.gpkg")
    gdfs = [gpd.read_file(f) for f in flist]
    pd.concat(gdfs).to_file("data/waterdistance/choptank.gpkg")
    [os.remove(f) for f in flist]

# --- pautexent
if not os.path.exists("data/waterdistance/pautexent.gpkg"):
    (start_idx_x, start_idx_y) = fwi.get_idx_coords(
        stations.iloc[[2]].reset_index()["longitude"][0],
        stations.iloc[[2]].reset_index()["latitude"][0],
        dt
    )
    # end_points.shape[0]  # 294049
    l = np.array_split(np.array(range(0, end_points.shape[0])), 6)
    l[-1] = np.delete(l[-1], np.array([11311]))  # resolve mystery coredump error
    # [s[-1] for s in l]
    [get_distance_grp(s) for s in l]

    flist = glob.glob("test_*.gpkg")
    gdfs = [gpd.read_file(f) for f in flist]
    pd.concat(gdfs).to_file("data/waterdistance/pautexent.gpkg")
    [os.remove(f) for f in flist]

# --- potomac
if not os.path.exists("data/waterdistance/potomac.gpkg"):
    (start_idx_x, start_idx_y) = fwi.get_idx_coords(
        stations.iloc[[3]].reset_index()["longitude"][0],
        stations.iloc[[3]].reset_index()["latitude"][0],
        dt
    )
    # end_points.shape[0]  # 294049
    l = np.array_split(np.array(range(0, end_points.shape[0])), 6)
    l[-1] = np.delete(l[-1], np.array([11311]))  # resolve mystery coredump error
    # [s[-1] for s in l]
    [get_distance_grp(s) for s in l]

    flist = glob.glob("test_*.gpkg")
    gdfs = [gpd.read_file(f) for f in flist]
    pd.concat(gdfs).to_file("data/waterdistance/potomac.gpkg")
    [os.remove(f) for f in flist]

# --- susquehanna
if not os.path.exists("data/waterdistance/susquehanna.gpkg"):
    (start_idx_x, start_idx_y) = fwi.get_idx_coords(
        stations.iloc[[1]].reset_index()["longitude"][0],
        stations.iloc[[1]].reset_index()["latitude"][0],
        dt
    )
    # end_points.shape[0]  # 294049
    l = np.array_split(np.array(range(0, end_points.shape[0])), 20)

    # fmt: off
    l[0] = np.delete(l[0], np.array([10170+156])) # resolve mystery coredump error
    l[4] = np.delete(l[4], np.array([2600+77])) # resolve mystery coredump error
    l[4] = np.delete(l[4], np.array([3100+116])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([1300+100])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([1300+100])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([1800+79])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([2100+56])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([2100+56])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([2200+63])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([2600+61])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([2900+11])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([3000+27])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([3500+17])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([8850+44])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([8850+44])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([9600+14])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10350+21])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10640+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([10900+25])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([11200+5])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([11450+49])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12100+10])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([12700+13])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([13000+9])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([13000+9])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([13000+9])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([13000+9])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([13300+24])) # resolve mystery coredump error    
    l[6] = np.delete(l[6], np.array([13600+47])) # resolve mystery coredump error
    l[6] = np.delete(l[6], np.array([14300+21])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([350+36])) # resolve mystery coredump error    
    l[7] = np.delete(l[7], np.array([720+21])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([1050+25])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([1400+15])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([1400+15])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([1400+15])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([1700+58])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([2050+40])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([2080+10])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([2400+32])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([2400+32])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([2400+32])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([2750+34])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([3140+4])) # resolve mystery coredump error
    l[7] = np.delete(l[7], np.array([3890+9])) # resolve mystery coredump error
    # fmt: on

    # [s[-1] for s in l]
    # random.shuffle(l)
    [get_distance_grp(s) for s in l]

    flist = glob.glob("test_*.gpkg")
    gdfs = [gpd.read_file(f) for f in flist]
    pd.concat(gdfs).to_file("data/waterdistance/susquehanna.gpkg")
    [os.remove(f) for f in flist]

# i = l[7][1700+58]
# get_distance(end_points[i][1], end_points[i][0], i)

# test = np.array(range(l[7][3890], l[7][len(l[7])-1]))
# res = []
# for i in tqdm(test):
#     # print(i)
#     res.append(get_distance(end_points[i][1], end_points[i][0], i))

discharge = pd.read_csv("data/discharge_median.csv")

flist = glob.glob("data/waterdistance/*.gpkg")


if not os.path.exists("data/waterdistance.tif"):
    grids = [fwi.weight_grid(f, discharge) for f in flist]

    merged = xr.concat(grids, dim="band")
    merged = merged.sum(dim="band", skipna=True)
    merged = merged.where(merged > 10)
    merged = merged.where(merged < 5000)
    merged = merged.bfill("time")

    merged.rio.to_raster("data/waterdistance.tif")

# extract cost for ALL locations including prediction locations
wd_raw = rasterio.open("data/waterdistance.tif")
pixel_centers = gpd.read_file("data/pixel_centers_4326.shp").rename(
    columns={"pix_idx": "pix_id"}
)

coord_list = [
    (x, y) for x, y in zip(pixel_centers["geometry"].x, pixel_centers["geometry"].y)
]

pixel_centers["cost"] = [x[0] for x in wd_raw.sample(coord_list)]
pixel_centers["longitude"] = [x[0] for x in coord_list]
pixel_centers["latitude"] = [x[1] for x in coord_list]
# pixel_centers.to_file("test.gpkg")
pixel_centers = pixel_centers.drop(columns=["geometry"])
pixel_centers.to_csv("data/fwi_cost.csv", index=False)


# join cost to measurement locations
aggregated_w_bandvals = pd.read_csv("data/aggregated_w_bandvals.csv")
res = aggregated_w_bandvals.merge(pixel_centers)
res_unique = (
    res.groupby(["latitude", "longitude", "pix_id", "cost"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
# gpd.GeoDataFrame(res_unique, geometry=gpd.points_from_xy(res_unique.longitude, res_unique.latitude)).to_file("test2.gpkg")
res.to_csv(
    "data/data_w_fwi.csv",
    index=False,
)

# ---
# # --- susq
# test_coords = []
# if not os.path.exists("susq.gpkg"):
#     (start_idx_x, start_idx_y) = get_idx_coords(stations.iloc[[1]].reset_index(drop=True)["longitude"][0],
#                                                 stations.iloc[[1]].reset_index(drop=True)["latitude"][0])
#     l = np.array_split(np.array(range(0, end_points.shape[0])), 6)

#     i = l[0][8162]
#     get_distance(end_points[i][1], end_points[i][0], i)

#     test_coords.append(end_points[l[0][8162]])
#     l[0] = np.delete(l[0], np.array([8162])) # resolve mystery coredump error
#     test_coords.append(end_points[l[0][9680]])
#     l[0] = np.delete(l[0], np.array([9680])) # resolve mystery coredump error
#     test_coords.append(end_points[l[0][9680]])
#     l[0] = np.delete(l[0], np.array([9680])) # resolve mystery coredump error
#     test_coords.append(end_points[l[0][20675]])
#     l[0] = np.delete(l[0], np.array([20675])) # resolve mystery coredump error
#     test_coords.append(end_points[l[0][20679]])
#     l[0] = np.delete(l[0], np.array([20679])) # resolve mystery coredump error
#     l[0] = np.delete(l[0], np.array([20678])) # resolve mystery coredump error
#     l[0] = np.delete(l[0], np.array([20678])) # resolve mystery coredump error
#     l[0] = np.delete(l[0], np.array([20678])) # resolve mystery coredump error
#     l[0] = np.delete(l[0], np.array([20678])) # resolve mystery coredump error

#     # get_idx_coords(*get_geo_coords(test_coords[0][0],test_coords[0][1]))
#     # test_coords_pd = pd.DataFrame([get_geo_coords(coords[1], coords[0]) for coords in end_points[0:1000]], columns=["x", "y"])
#     test_coords_pd = pd.DataFrame([get_geo_coords(coords[1], coords[0]) for coords in test_coords], columns=["x", "y"])
#     gpd.GeoDataFrame(geometry=gpd.points_from_xy(test_coords_pd.x, test_coords_pd.y)).to_file("test3.gpkg", driver="GPKG")


#     # i = l[0][0]
#     # get_distance(end_points[i][1], end_points[i][0], i)

#     # [s[-1] for s in l]
#     [get_distance_grp(s) for s in l]

#     flist = glob.glob("test_*.gpkg")
#     gdfs = [gpd.read_file(f) for f in flist]
#     pd.concat(gdfs).to_file("susq.gpkg")
#     [os.remove(f) for f in flist]

# test = np.array(range(20675, l[0][len(l[0])-1]))
# # plus 3
# i = 20676 + 2
# i = l[0][len(l[0])-1]
# get_distance(end_points[i][1], end_points[i][0], i)

# # i = l[-1][11311]
# # get_distance(end_points[i][1], end_points[i][0], i)
# # (end_points[i][1], end_points[i][0])
# # get_geo_coords(end_points[i][1], end_points[i][0])
