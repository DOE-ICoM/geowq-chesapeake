import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from src import utils

variable = "salinity"
best_params = utils.load_md(variable)

# read in a single year of GEE data
predictors = [
    'datetime', 'sur_refl_b08', 'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11',
    'sur_refl_b12', 'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15',
    'sur_refl_b16', 'latitude', 'longitude'
]
X_predict = pd.read_csv(
    os.environ["ICOM_DATA"] +
    "/Modeling Data/Processed Data p1/aggregated_w_bandvals.csv").dropna()
X_predict["datetime"] = utils.datetime_to_doy(X_predict["datetime"])

test = np.array(X_predict[predictors])
# calculate 'Ratio 1', 'Ratio 2', 'Ratio 3' ?
X_train = pickle.load(open("data/X_train_" + variable + ".pkl", "rb"))
# X_train.shape
X_train[0]

rf_random_path = "data/rf_random_" + variable + ".pkl"
rf_random = pickle.load(open(rf_random_path, "rb"))
predictions = rf_random.predict(test)

res = X_predict[["SSS (psu)"]].copy()
res["predict"] = predictions.copy()

sns.scatterplot(data=res, x="predict", y="SSS (psu)")
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