import sys
import subprocess
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as anim

sys.path.append(".")
from src import utils


def add_day(date, n=1):
    date_plus_n = str(datetime.strptime(date, "%Y-%m-%d") +
                      timedelta(days=n)).split(" ")[0]
    return date_plus_n


def index_xr(x, date):
    # x = imgs_rf[0]
    # date = dates[0]
    date_pd = pd.date_range(date, date, freq='D', inclusive='left')
    date_xr = xr.DataArray(date_pd, [("time", date_pd)])
    x = x.expand_dims(time=date_xr)
    return x


n_days = 61

start_date = "2011-07-31"
# start_date = "2012-07-31"
dates = [add_day(start_date, i + 1) for i in range(n_days)]
# [
#     subprocess.call('python scripts/03_get_data_predict.py --date ' + date)
#     for date in dates
# ]
# [
#     subprocess.call('python scripts/04_rf_predict.py --date ' + date +
#                     ' --variable salinity --var_col "SSS (psu)"')
#     for date in dates
# ]

imgs_rf = [
    utils.get_rf_prediction(date, "salinity", smoothing=True) for date in dates
]
imgs_rf = [index_xr(x, date) for x, date in zip(imgs_rf, dates)]

ts = pd.DataFrame({
    "salinity":
    [float(imgs_rf[i].median().values) for i in range(len(imgs_rf))],
    "date":
    pd.to_datetime(dates, infer_datetime_format=True)     
})
g = sns.lineplot(data=ts, x="date", y="salinity")
g.get_figure().autofmt_xdate()
plt.show()



test = xr.concat(imgs_rf, dim="time")
variable = test.sel(time=slice('2011', '2012'))


def update(t):
    # t = variable.time.values[0]
    print(t)
    ax.set_title("time = %s" % t)
    image.set_array(variable.sel(time=t).squeeze())
    return image,


fig = plt.figure(figsize=plt.figaspect(1.22))
fig.subplots_adjust(0.1, 0, 1, 1)
ax = plt.axes(projection=ccrs.PlateCarree())
image = variable.isel(time=0).squeeze().plot.imshow(
    ax=ax, transform=ccrs.PlateCarree(), animated=True)

animation = anim.FuncAnimation(fig,
                               update,
                               interval=400,
                               frames=variable.time.values,
                               blit=False)

plt.box(False)
ax.annotate('Salinity',
            xy=(0, 0),
            xycoords='figure fraction',
            xytext=(0.04, 0.97),
            textcoords='figure fraction',
            fontsize=14)
animation.save("irene.gif")
# plt.show()