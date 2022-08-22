# https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html

# data products are every 6 hours

import rioxarray
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

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

test = ds1.sel({"Depth":0}).isel(ocean_time=[0]).drop_vars(["Depth"])
test = test.salt.to_dataset()

test.salt[0,:,:].plot.imshow()
plt.show()

# test.to_netcdf("salt.nc")
