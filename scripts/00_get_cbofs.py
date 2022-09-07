# https://tidesandcurrents.noaa.gov/ofs/cbofs/cbofs.html
# data products are every 6 hours
# data products start from the current date
#  and go back one month in time

# "regulargrid" files are only available for the prior 2 days...
# https://github.com/pydata/xarray/issues/6453
# https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/2022/09/07/nos.cbofs.regulargrid.n001.20220907.t00z.nc

import os
import argparse
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tod", type=str)
    args = vars(parser.parse_args())
    tod = args["tod"]

    # tod = "20220904"
    tif_path = "data/cbofs/salt_{date}.tif".format(date=tod)

    if not os.path.exists(tif_path):
        nc_path = 'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CBOFS/MODELS/{year}/{month}/{day}/nos.cbofs.regulargrid.n001.{date}.t00z.nc'.format(date=tod, year=tod[0:4], month=tod[4:6], day=tod[6:8])

        # print(nc_path)
        ds1 = xr.open_dataset(nc_path)

        # ---        
        # nc_path = 'https://www.ncei.noaa.gov/thredds/dodsC/model-cbofs-files/2022/09/nos.cbofs.fields.n001.20220904.t00z.nc'
        # ds1 = xr.open_dataset(nc_path)
        # data = ds1["salt"].as_numpy()[0,0,:,:].values
        # lons = ds1.lon_rho.as_numpy()[0].values
        # lats = ds1.lat_rho.as_numpy()[:,0].values
        # data = xr.DataArray(data=data, dims=["y", "x"], coords=dict(x=(lons), y=(lats)))
        # data = data.rio.set_spatial_dims("x", "y")
        # data.rio.write_crs(4326, inplace=True)
        # data.rio.to_raster("test.tif")        
        # ---

        dt = ds1.sel({"Depth": 0}).isel(ocean_time=[0]).drop_vars(["Depth"])
        dt = dt.salt.to_dataset()
        # dt.salt[0,:,:].plot.imshow()
        # plt.show()

        # re-grid to avoid the ocean time dimension
        lons = dt.Longitude.as_numpy().values[0]
        lats = dt.Latitude.as_numpy().values[:,0]
        data = dt["salt"].as_numpy().values[0, :, :]

        data = xr.DataArray(data=data, dims=["y", "x"], coords=dict(x=(lons), y=(lats)))
        data = data.rio.set_spatial_dims("x", "y")
        data.rio.write_crs(4326, inplace=True)

        data.rio.to_raster(tif_path)

if __name__ == "__main__":
    main()