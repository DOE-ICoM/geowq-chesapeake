# https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html

# data products are every 6 hours

# netcdf on aws

import pandas as pd
import xarray as xr

tod = pd.Timestamp.today()
locs = [
    tod.strftime(
        'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/%Y/%m/%d/nos.cbofs.regulargrid.n001.%Y%m%d.t00z.nc'
    ),
    tod.strftime(
        'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/%Y/%m/%d/nos.cbofs.regulargrid.n002.%Y%m%d.t00z.nc'
    )
]

# THIS WORKS: using `open_mfdataset` with 1 file
ds1 = xr.open_mfdataset([locs[0]])
print('DATASET1: ', ds1['ocean_time'].attrs, ds1['ocean_time'].encoding)
print(ds1['ocean_time'].dtype)
print(xr.decode_cf(ds1).ocean_time.encoding)
xr.decode_cf(ds1).to_netcdf('test1.nc')