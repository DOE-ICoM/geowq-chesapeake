import sys
import pickle
import xarray as xr
import pandas as pd
import geopandas as gpd

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

(start_idx_x,
 start_idx_y) = fwi.get_idx_coords(stations.iloc[[0]]["longitude"][0],
                                   stations.iloc[[0]]["latitude"][0], dt)

end_points = pickle.load(open("data/end_points.pkl", "rb"))

i = 171133
# i=45050
test = fwi.get_distance(end_points[i][1],
                 end_points[i][0],
                 i,
                 start_idx_y,
                 start_idx_x,
                 cost_surface_array,
                 dt,
                 return_full_path=True)

test.to_file("test.gpkg", driver="GPKG")

discharge = pd.read_csv("data/discharge_median.csv")