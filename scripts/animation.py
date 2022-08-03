import glob
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as anim

flist = glob.glob("data/prediction/*.tif")


def read_xr(date):
    # date = "2018-01-01"
    fpath = "data/prediction/" + date + ".tif"
    date_pd = pd.date_range(date, date, freq='D', inclusive='left')
    date_xr = xr.DataArray(date_pd, [("time", date_pd)])

    dt = xr.open_dataset(fpath, engine="rasterio")
    dt = dt.expand_dims(time=date_xr)
    return dt


test = xr.concat([read_xr("2018-01-01"), read_xr("2018-01-02")], dim="time")

variable = test.band_data.sel(time=slice('2018', '2018'))

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())

image = variable.isel(time = 0).squeeze().plot.imshow(ax=ax, transform=ccrs.PlateCarree(), animated=True)

def update(t):    
    # t = variable.time.values[0]
    print(t)
    ax.set_title("time = %s"%t)
    image.set_array(variable.sel(time=t).squeeze())
    return image,

animation = anim.FuncAnimation(fig, update, frames=variable.time.values, blit=False)

#animation.save('tasmax.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
