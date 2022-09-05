# https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html
# data products are every 6 hours
# data products seem to be archived for only a month in time

import os
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

tod = "20220904"
tif_path = "data/cbofs/salt_{date}.tif".format(date=tod)

if not os.path.exists(tif_path):
    locs = [
        'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/{year}/{month}/{day}/nos.cbofs.regulargrid.n001.{date}.t00z.nc'
        .format(date=tod, year=tod[0:4], month=tod[4:6], day=tod[6:8]),
        'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/%Y/%m/%d/nos.cbofs.regulargrid.n002.{date}.t00z.nc'
        .format(date=tod)
    ]
    ds1 = xr.open_dataset(locs[0])
    dt = ds1.sel({"Depth": 0}).isel(ocean_time=[0]).drop_vars(["Depth"])
    dt = dt.salt.to_dataset()

    # dt.salt[0,:,:].plot.imshow()
    # plt.show()

    # ---

    lons = dt.Longitude.as_numpy().values[0]
    lats = dt.Latitude.as_numpy().values[:,0]
    data = dt["salt"].as_numpy().values[0, :, :]

    test = xr.DataArray(data=data, dims=["y", "x"], coords=dict(x=(lons), y=(lats)))
    test = test.rio.set_spatial_dims("x", "y")
    test.rio.write_crs(4326, inplace=True)

    test.rio.to_raster(tif_path)
