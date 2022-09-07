import re
import glob
import rasterio
import datetime
import itertools
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

flist = glob.glob("data/prediction/*.tif")
flist = list(
    itertools.compress(
        flist, [not strng.__contains__("downsample") for strng in flist]))
flist = list(
    itertools.compress(flist,
                       [not strng.__contains__("2018") for strng in flist]))

dates = [x.split("\\")[1].replace(".tif", "") for x in flist]

pixel_centers_all = gpd.read_file("data/pixel_centers_4326.shp").rename(
    columns={"pix_idx": "pix_id"})

def _extract(i):
    wd_raw = rasterio.open(flist[i])
    pixel_centers = pixel_centers_all[[
        x in [270359986, 272951937] for x in pixel_centers_all["pix_id"]
    ]].copy()
    coord_list = [(x, y) for x, y in zip(pixel_centers["geometry"].x,
                                         pixel_centers["geometry"].y)]
    pixel_centers["salt"] = [x[0] for x in wd_raw.sample(coord_list)]
    pixel_centers["date"] = dates[i]
    return pixel_centers


res = [_extract(i) for i in range(0, len(dates))]
res = pd.concat(res).reset_index(drop=True)

# ---

sns.lineplot(x="date", y="salt", data=res)
plt.show()