import glob
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def read_xr(date):
    # date = "2018-01-01"
    fpath = "data/save_cbofs/salt_" + date + ".tif"
    date_pd = pd.date_range(date, date, freq='D', inclusive='left')
    date_xr = xr.DataArray(date_pd, [("time", date_pd)])

    dt = xr.open_dataset(fpath, engine="rasterio")
    dt = dt.expand_dims(time=date_xr)
    return dt


def f_to_date(x):
    # x = flist[0]
    return x.split("_")[2].replace(".tif", "")
    # return x[0:4] + "-" + x[4:6] + "-" + x[6:8]


flist = glob.glob("data/save_cbofs/*.tif")
dates = [f_to_date(f) for f in flist]

#  --- animation
xr_list = [read_xr(date) for date in dates]
test = xr.concat(xr_list, dim="time")

variable = test.band_data.sel(time=slice('2021', '2022'))

def update(t):
    # t = variable.time.values[0]
    print(t)
    ax.set_title("time = %s" % t)
    image.set_array(variable.sel(time=t).squeeze())
    return image,

fig = plt.figure(figsize=plt.figaspect(1.22))
fig.subplots_adjust(0.1,0,1,1)
ax = plt.axes(projection=ccrs.PlateCarree())
image = variable.isel(time=0).squeeze().plot.imshow(
    ax=ax, transform=ccrs.PlateCarree(), animated=True)

animation = anim.FuncAnimation(fig,
                               update,
                               interval=400,
                               frames=variable.time.values,
                               blit=False)

plt.box(False)
ax.annotate('Salinity', xy = (0, 0), xycoords='figure fraction',
            xytext=(0.04, 0.97), textcoords='figure fraction', fontsize=14
            )
animation.save("cbofs.gif")
# plt.show()

# --- simple plotting
test = read_xr("2018-01-01")
da = test.isel(band=0).band_data.expand_dims({"band": 1})

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
image = da.squeeze().plot.imshow(ax=ax,
                                 transform=ccrs.PlateCarree(),
                                 animated=True)
plt.show(block=False)