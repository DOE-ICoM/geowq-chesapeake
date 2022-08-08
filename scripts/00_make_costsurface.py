import numpy as np
import xarray as xr
import geopandas as gpd
from rivgraph import im_utils
from geocube.api.core import make_geocube

chk = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(4326)
chk_hull = gpd.GeoDataFrame(geometry=chk.geometry.convex_hull)
chk_hull["cost"] = 1000

# rasterize convex hull
cost_grid = make_geocube(
    vector_data=chk_hull,
    measurements=['cost'],
    resolution=(-0.002, 0.002),
)
# cost_grid["cost"].rio.to_raster("cost.tif")

# rasterize chk_water_only
chk["geometry"] = max(chk.geometry[0], key=lambda a: a.area)
chk["cost"] = -999
cost_grid_water = make_geocube(
    vector_data=chk,
    measurements=['cost'],
    resolution=(-0.002, 0.002),
)

# remove slivers
cost_grid_water_clean = im_utils.regionprops(
    np.nan_to_num(cost_grid_water.cost.values),
    props=["area"],
)[1]
cost_grid_water_clean[cost_grid_water_clean == 1] = -999
cost_grid_water_clean[cost_grid_water_clean > 1] = 0
cost_grid_water_clean = cost_grid_water_clean.astype('float64')
cost_grid_water_clean = xr.DataArray(
    cost_grid_water_clean,
    coords=[cost_grid_water.y.values, cost_grid_water.x.values],
    dims=["y", "x"])
cost_grid_water_clean["spatial_ref"] = cost_grid_water["spatial_ref"]
cost_grid_water_clean = cost_grid_water_clean.to_dataset(name="cost")

# raster math to create high land cost, low water cost
cost = xr.concat(
    [cost_grid.expand_dims(band=1),
     cost_grid_water_clean.expand_dims(band=1)],
    dim="band")
cost = cost.sum(dim="band", skipna=True)
cost = cost.where(cost > 0)
cost = cost.bfill('time')

cost["cost"].rio.to_raster("data/cost.tif")
