import xarray as xr
import geopandas as gpd
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
chk["cost"] = -999
cost_grid_water = make_geocube(
    vector_data=chk,
    measurements=['cost'],
    resolution=(-0.002, 0.002),
)
# cost_grid_water["cost"].rio.to_raster("cost_water.tif")

# raster math to create high land cost, low water cost
cost = xr.concat(
    [cost_grid.expand_dims(band=1),
     cost_grid_water.expand_dims(band=1)],
    dim="band")
cost = cost.sum(dim="band", skipna=True)
cost = cost.where(cost > 0)
cost = cost.bfill('time')

cost["cost"].rio.to_raster("data/cost.tif")
