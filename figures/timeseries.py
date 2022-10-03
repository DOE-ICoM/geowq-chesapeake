import glob
import rasterio
import itertools
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

flist = glob.glob("data/prediction/*.tif")
flist = list(
    itertools.compress(
        flist, [not strng.__contains__("downsample") for strng in flist]))
flist = list(
    itertools.compress(flist,
                       [not strng.__contains__("2018") for strng in flist]))
flist = list(
    itertools.compress(flist,
                       [strng.__contains__("salinity") for strng in flist]))

dates = [x.split("\\")[1].replace(".tif", "").replace("_salinity", "") for x in flist]

pixel_centers_all = gpd.read_file("data/pixel_centers_4326.shp").rename(
    columns={"pix_idx": "pix_id"})
pixel_centers = pixel_centers_all.sample(100)


def _extract(i):
    wd_raw = rasterio.open(flist[i])
    pixel_centers_i = pixel_centers.copy()
    coord_list = [(x, y) for x, y in zip(pixel_centers["geometry"].x,
                                         pixel_centers["geometry"].y)]
    pixel_centers_i["salt"] = [x[0] for x in wd_raw.sample(coord_list)]
    pixel_centers_i["date"] = dates[i]
    return pixel_centers_i


res_raw = [_extract(i) for i in range(0, len(dates))]
res = pd.concat(res_raw).reset_index(drop=True)
res["date"] = [x.replace("_salinity", "") for x in res["date"]]
res["date"] = pd.to_datetime(res["date"])
res["month"] = [x.strftime("%m") for x in res["date"]]
x_dates = res['date'].dt.strftime('%m').sort_values().unique()

months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b')
fig, ax = plt.subplots(figsize=(12, 6))
fig = sns.lineplot(x="date", y="salt", data=res)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
plt.xlabel("")
plt.ylabel("Salinity")

# plt.show()
plt.savefig("figures/_annual-cycle.pdf")
