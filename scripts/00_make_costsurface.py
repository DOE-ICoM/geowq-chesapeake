import geopandas as gpd
from geocube.api.core import make_geocube

stations = gpd.GeoDataFrame({
    "name": ['Choptank', 'Susquehanna', 'Pautexent', 'Potomac'],
    "longitude": [-75.900, -76.083, -76.698, -77.059],
    "latitude": [38.811, 39.553, 38.8311, 39.379]
})
stations = gpd.GeoDataFrame(stations,
                            geometry=gpd.points_from_xy(
                                stations["longitude"], stations["latitude"]))

chk = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(4326)
chk_hull = gpd.GeoDataFrame(geometry=chk.geometry.convex_hull)
chk_hull["cost"] = 1000

# rasterize convex hull

cost_grid = make_geocube(
    vector_data=chk_hull,
    measurements=['cost'],
    resolution=(-0.002, 0.002),
)

cost_grid["cost"].rio.to_raster("cost.tif")

# rasterize chk_water_only

# raster math to create high land cost, low water cost

# save