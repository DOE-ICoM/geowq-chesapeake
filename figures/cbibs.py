import os
import json
import requests
import itertools
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

outfile = "data/cbibs.csv"

## CBIBS
if not os.path.exists(outfile):
    base_url = 'https://mw.buoybay.noaa.gov/api/v1'
    apikey = 'f159959c117f473477edbdf3245cc2a4831ac61f'
    start = '2000-09-08T01:00:00z'
    end = '2021-12-09T23:59:59z'
    var = 'Position'

    query_url = '{}/json/query?key={}&sd={}&ed={}&var={}'.format(base_url,apikey,start,end,var)

    pull = json.loads(requests.get(query_url).text)

    def _extract(dt):
        # dt = pull["stations"][0]
        if dt["stationLongName"].__contains__("CBIBS") and dt["active"]:
            coords = {k: v for k, v in dt.items() if k.endswith('tude')}
            name = {k: v for k, v in dt.items() if k.endswith('ShortName')}
            res = name | coords
        else:
            res = None
        return res

    stations = [_extract(dt) for dt in pull["stations"]]
    is_cbibs = [stations[i] is not None for i in range(0, len(stations))]
    stations = list(itertools.compress(stations, is_cbibs))

    stations = pd.DataFrame(stations)
    stations.to_csv(outfile, index=False)

stations = pd.read_csv(outfile)
stations = gpd.GeoDataFrame(stations,
                               geometry=gpd.points_from_xy(
                                   stations["longitude"], stations["latitude"]))

b = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(
    epsg=4326)
b_bounds = [x for x in b.bounds.iloc[0]]
extent = (b_bounds[0], b_bounds[2], b_bounds[1], b_bounds[3])

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent, ccrs.PlateCarree())
ax.coastlines(resolution="10m", color="black", linewidth=1)
stations.plot(ax=ax)
stations.apply(lambda x: ax.annotate(text=x['stationShortName'], xy=x.geometry.coords[0], xytext=(3, 3), textcoords="offset points"), axis=1)
plt.savefig("figures/cbibs.pdf")