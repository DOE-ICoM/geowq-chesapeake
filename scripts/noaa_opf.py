# https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html

# data products are every 6 hours
# data products seem to be archived for only a month in time

import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

tod = "20220904"
locs = [
    'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/{year}/{month}/{day}/nos.cbofs.regulargrid.n001.{date}.t00z.nc'.format(date=tod, year=tod[0:4], month=tod[4:6], day=tod[6:8]),
    'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/%Y/%m/%d/nos.cbofs.regulargrid.n002.{date}.t00z.nc'.format(date=tod)    
]

ds1 = xr.open_dataset(locs[0])

test = ds1.sel({"Depth":0}).isel(ocean_time=[0]).drop_vars(["Depth"])
test = test.salt.to_dataset()

test.salt[0,:,:].plot.imshow()
plt.show()

# test.to_netcdf("salt.nc")
test = xr.open_dataset("salt.nc")
