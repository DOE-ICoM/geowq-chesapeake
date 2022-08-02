import os
import pickle
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

from functools import partial
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata

from src import utils
from src import rf_icom_call_data2 as call_data2


def _get_coords(x):
    x = x.split("[")[1]
    longitude = float(x.split(",")[0])
    latitude = float(x.split(",")[1].split("]")[0])
    return (longitude, latitude)


variable = "salinity"
var_col = "SSS (psu)"
best_params = utils.load_md(variable)

rf_random_path = "data/rf_random_" + variable + ".pkl"
rf_random = pickle.load(open(rf_random_path, "rb"))

# read in a single year of GEE data
predictors = [
    'datetime', 'sur_refl_b08', 'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11',
    'sur_refl_b12', 'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15',
    'sur_refl_b16', 'latitude', 'longitude'
]

# --- prep prediction data
test = pd.read_csv("data/prediction/modis-2018_01_01.csv").drop(
    columns=["system:index"])
test["datetime"] = '2018-01-01 00:00:00'
test[var_col] = -99
datetime = test.pop("datetime")
test.insert(0, "datetime", datetime)

longs = []
lats = []
for i in range(0, test.shape[0]):
    coords = _get_coords(test[".geo"][i])
    longs.append(coords[0])
    lats.append(coords[1])

test["longitude"] = longs
test["latitude"] = lats

longitude = test.pop("longitude")
test.insert(1, "longitude", longitude)
latitude = test.pop("latitude")
test.insert(2, "latitude", latitude)

test = call_data2.clean_data(variable,
                             var_col,
                             predictors,
                             test_size=0,
                             data=test)

# --- predict prediction data
predictions = rf_random.predict(test[0])
res = pd.DataFrame(test[1], columns=["obs"])
res["predict"] = predictions.copy()
res["longitude"] = longs
res["latitude"] = lats
res = gpd.GeoDataFrame(data=res,
                       geometry=gpd.points_from_xy(res.longitude,
                                                   res.latitude))

geo_grid = make_geocube(
    vector_data=res,
    measurements=['predict'],
    resolution=(-0.1, 0.00001),    
    rasterize_function=partial(rasterize_points_griddata, filter_nan=True),
)

geo_grid.predict.where(geo_grid.predict != geo_grid.predict.rio.nodata).plot()
plt.show()

geo_grid["predict"].rio.to_raster("my_rasterized_column.tif")

# res.to_file("test.gpkg", driver="GPKG")
# sns.histplot(data=res, x="predict")
# plt.show()

# --- prep validation data
X_predict_raw = pd.read_csv(
    os.environ["ICOM_DATA"] +
    "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv").dropna()
X_predict = call_data2.clean_data(variable,
                                  var_col,
                                  predictors,
                                  test_size=0,
                                  data=X_predict_raw)

# calculate 'Ratio 1', 'Ratio 2', 'Ratio 3' ?
# X_train = pickle.load(open("data/X_train_" + variable + ".pkl", "rb"))
# X_train.shape
# X_train[0]

# --- predict validation data
predictions = rf_random.predict(X_predict[0])
res = pd.DataFrame(X_predict[1], columns=["obs"])
res["predict"] = predictions.copy()

sns.scatterplot(data=res, x="predict", y="obs")
plt.show()

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