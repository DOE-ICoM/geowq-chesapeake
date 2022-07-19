import re
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor

variable = "salinity"


def _unwrap(y):
    return [[x for x in re.split('\s+', y)]]


# load best_params_[variable].md
bp_colnames = _unwrap(
    str(
        pd.read_table("data/best_params_" + variable + ".md",
                      nrows=1,
                      header=None)[0][0]).strip())[0]
bp_colnames.remove("max_leaf_nodes")
bp = pd.DataFrame(_unwrap(
    str(
        pd.read_table("data/best_params_" + variable + ".md",
                      skiprows=2,
                      header=None)[0][0]).strip()),
                  columns=bp_colnames)

# read in a single year of GEE data
predictors = [
    'datetime', 'sur_refl_b08', 'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11',
    'sur_refl_b12', 'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15',
    'sur_refl_b16', 'latitude', 'longitude'
]
X_predict = pd.read_csv(
    os.environ["ICOM_DATA"] +
    "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv")
X_predict.head()

test = X_predict.groupby(['loc_id', "latitude",
                          "longitude"]).size().reset_index().rename(columns={
                              0: 'count'
                          }).sort_values("count", ascending=False).reset_index(drop=True)
test.head()
test2 = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test["longitude"], test["latitude"]))
test2.to_file("test2.gpkg", driver="GPKG")

# TODO: groupby loc_id and see if all locs have the same frequency counts
test = X_predict.groupby(["loc_id"]).size().reset_index()

test = np.array(X_predict[predictors])
# calculate 'Ratio 1', 'Ratio 2', 'Ratio 3' ?
X_train = pickle.load(open("data/X_train_" + variable + ".pkl", "rb"))
# X_train.shape

ols_path = "data/ols_" + variable + ".pkl"
ols = pickle.load(open(ols_path, "rb"))

predictions = ols.predict(X_predict)

# .predict()