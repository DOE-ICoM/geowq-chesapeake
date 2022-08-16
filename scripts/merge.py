import glob
import xarray as xr
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube

discharge = pd.read_csv("data/discharge_median.csv")

flist = glob.glob("*.gpkg")
flist.remove("stations.gpkg")


def weight_grid(f):
    # f = flist[0]
    site_str = f.replace(".gpkg", "").title()

    gdf = gpd.read_file(f)
    cost_grid = make_geocube(
        vector_data=gdf,
        measurements=['cost'],
        resolution=(-0.002, 0.002),
    )

    discharge_site = float(
        discharge[discharge["site_str"] == site_str]["discharge_va"])

    return (discharge_site / cost_grid).expand_dims(band=1)


grids = [weight_grid(f) for f in flist]

merged = xr.concat(grids, dim="band")
merged = merged.sum(dim="band", skipna=True)
merged = merged.where(merged > 0)
merged = merged.bfill('time')

merged.rio.to_raster("merged2.tif")