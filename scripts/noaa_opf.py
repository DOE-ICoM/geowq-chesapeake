# https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html

# data products are every 6 hours

import pandas as pd
import xarray as xr
import rioxarray

tod = pd.Timestamp.today()
locs = [
    tod.strftime(
        'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/%Y/%m/%d/nos.cbofs.regulargrid.n001.%Y%m%d.t00z.nc'
    ),
    tod.strftime(
        'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/%Y/%m/%d/nos.cbofs.regulargrid.n002.%Y%m%d.t00z.nc'
    )
]

ds1 = xr.open_dataset(locs[0])

test = ds1.sel({"Depth":0})
test = test.isel(ocean_time=[0])
test = test.drop_vars(["Depth"])

test.salt.to_dataset().to_netcdf("salt.nc")
