import glob
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

flist = glob.glob("data/prediction/*.tif")

test = xr.open_dataset(flist[0], engine="rasterio")
test2 = xr.open_dataset(flist[1], engine="rasterio")

da = test.isel(band=0).band_data.expand_dims({"band": 1})

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
image = da.squeeze().plot.imshow(ax=ax,
                                 transform=ccrs.PlateCarree(),
                                 animated=True)
plt.show()
