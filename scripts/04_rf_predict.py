import os
import sys
import pickle
import argparse
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

from functools import partial
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata

sys.path.append(".")
from src import fit_sine
from src import rf_icom_call_data2 as call_data2


def _get_coords(x):
    x = x.split("[")[1]
    longitude = float(x.split(",")[0])
    latitude = float(x.split(",")[1].split("]")[0])
    return (longitude, latitude)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--variable", type=str)
    parser.add_argument("--var_col", type=str)
    args = vars(parser.parse_args())
    date = args["date"]
    variable = args["variable"]
    var_col = args["var_col"]
    # date = "2022-09-04"
    # variable = "salinity"
    # var_col = "SSS (psu)"

    rf_random_path = "data/rf_random_" + variable + ".pkl"
    rf_random = pickle.load(open(rf_random_path, "rb"))

    predictors = pickle.load(open("data/imp_params_" + variable + ".pkl", "rb"))
    # predictors = [
    #     'datetime', 'sur_refl_b08', 'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11',
    #     'sur_refl_b12', 'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15',
    #     'sur_refl_b16', 'latitude', 'longitude', "cost"
    # ]

    # --- prep prediction data
    dt = pd.read_csv("data/prediction/modis-" + date.replace("-", "_") +
                     ".csv").drop(columns=["system:index"])
    dt["datetime"] = date + ' 00:00:00'
    dt[var_col] = -99
    datetime = dt.pop("datetime")
    dt.insert(0, "datetime", datetime)

    longs = []
    lats = []
    for i in range(0, dt.shape[0]):
        coords = _get_coords(dt[".geo"][i])
        longs.append(coords[0])
        lats.append(coords[1])

    dt["longitude"] = longs
    dt["latitude"] = lats

    longitude = dt.pop("longitude")
    dt.insert(1, "longitude", longitude)
    latitude = dt.pop("latitude")
    dt.insert(2, "latitude", latitude)

    # add fwi data
    fwi = pd.read_csv("data/fwi_cost.csv").drop(columns=["latitude", "longitude"])
    dt_raw = dt.merge(fwi, left_on="pix_idx", right_on="pix_id")

    dt_clean_X, dt_clean_y, lon_list, lat_list = call_data2.clean_data(variable,
                               var_col,
                               predictors,
                               test_size=0,
                               data=dt_raw)

    # --- predict prediction data
    predictions = rf_random.predict(dt_clean_X)
    res = pd.DataFrame(dt_clean_y, columns=["obs"])

    if variable == "temperature":
        # back out non-sine adjusted values    
        # pickle.load(open("data/feature_names_temperature.pkl", "rb"))
        p1 = pickle.load(open("data/temperature_sine_coef.pkl", "rb"))
        offset = fit_sine.fitfunc(p1, dt_clean_X[0:, 0])
        res["predict"] = predictions.copy() + offset
        res["obs"] = res["obs"] + offset
    else:
        res["predict"] = predictions.copy()

    res["longitude"] = lon_list
    res["latitude"] = lat_list
    res = gpd.GeoDataFrame(data=res,
                           geometry=gpd.points_from_xy(res.longitude,
                                                       res.latitude))

    # res.to_file("test3.gpkg", driver="GPKG")
    # sns.histplot(data=res, x="predict")
    # plt.show()

    geo_grid = make_geocube(
        vector_data=res,
        measurements=['predict'],
        resolution=(-0.002, 0.002),
        rasterize_function=partial(rasterize_points_griddata, filter_nan=True),
    )

    # bay_gdf = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(
    #     4326).buffer(0.02)
    # bay_gdf.to_file("data/Boundaries/bay_gdf.gpkg", driver="GPKG")
    bay_gdf = gpd.read_file("data/Boundaries/bay_gdf.gpkg")
    geo_grid = geo_grid.rio.clip(bay_gdf.geometry)

    # geo_grid.predict.where(
    #     geo_grid.predict != geo_grid.predict.rio.nodata).plot()
    # plt.show()

    out_path = "data/prediction/" + date + "_" + variable + ".tif"
    geo_grid["predict"].rio.to_raster(out_path)

    return out_path

if __name__ == "__main__":
    main()

# ---- parking lot

# test = X_predict.groupby(
#     ['loc_id', "latitude", "longitude"]).size().reset_index().rename(columns={
#         0: 'count'
#     }).sort_values("count", ascending=False).reset_index(drop=True)
# test.head()
# test2 = gpd.GeoDataFrame(test,
#                          geometry=gpd.points_from_xy(test["longitude"],
#                                                      test["latitude"]))
# test2.to_file("test2.gpkg", driver="GPKG")

# test = X_predict.groupby(["loc_id"]).size().reset_index()

# ---