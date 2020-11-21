# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:52:31 2020

@author: Jon

env:streamflow
"""
from erddapy import ERDDAP
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from pyproj import CRS

path_boundary = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Boundaries\delaware_chesapeake.shp"
gdf_bdry = gpd.read_file(path_boundary)
bb = gdf_bdry.geometry.values[0].bounds

e = ERDDAP(
  server='https://gliders.ioos.us/erddap',
  protocol='tabledap',
)

# Get all possible dataset_ids
kw = {
    "standard_name": "sea_water_temperature",
    "min_lon": bb[0],
    "max_lon": bb[2],
    "min_lat": bb[1],
    "max_lat": bb[3],
    "min_time": "1984-01-10T00:00:00Z",
    "max_time": "2020-12-31T00:00:00Z",
    "cdm_data_type": "trajectoryprofile"
}
search_url = e.get_search_url(response="csv", **kw)
search = pd.read_csv(search_url)
ds_ids = search['Dataset ID'].values

# Determine if any of the gliders are within the bays
has_data = []
errs = {}
for dsid in ds_ids:
    e.response = 'csv'
    e.dataset_id = dsid
    e.constraints = {
        'time>=': "1984-01-10T00:00:00Z",
        'time<=': "2020-12-31T00:00:00Z",
        'latitude>=': bb[1],
        'latitude<=': bb[3],
        'longitude>=': bb[0],
        'longitude<=': bb[2],
    }
    e.variables = [
        # 'depth',
        'latitude',
        'longitude',
        # 'salinity',
        # 'temperature',
        # 'time',
    ]
    url = e.get_download_url()
    try:
        df = e.to_pandas()
    except requests.HTTPError as exception:
        errs[dsid] = e.get_download_url()
        continue
        
    
    # Determine if any of the points are within our polygon
    for lat, lon in zip(df['latitude (degrees_north)'].values, df['longitude (degrees_east)'].values):
        pt = Point(lon, lat)
        if pt.intersects(gdf_bdry.geometry.values[0]) is True:
           has_data.append(dsid)
           print(dsid)
           break


# Get the glider data within the polygon
for hd in has_data:
    
    e.response = 'csv'
    e.dataset_id = hd
    e.constraints = {
        'time>=': "1984-01-10T00:00:00Z",
        'time<=': "2020-12-31T00:00:00Z",
        'latitude>=': bb[1],
        'latitude<=': bb[3],
        'longitude>=': bb[0],
        'longitude<=': bb[2],
    }
    e.variables = [
        'depth',
        'latitude',
        'longitude',
        'salinity',
        'temperature',
        'time',
    ]
    url = e.get_download_url()
    df = e.to_pandas()

    # Determine if any of the points are within our polygon
    geoms = []
    for lat, lon in zip(df['latitude (degrees_north)'].values, df['longitude (degrees_east)'].values):
        geoms.append(Point(lon, lat))
        
    gdf_e = gpd.GeoDataFrame(geometry=geoms, crs=CRS.from_epsg(4326))
    gdf_e.to_file(r'C:\Users\Jon\Desktop\Research\ICoM\Data\Gliders\{}.json'.format(hd), driver='GeoJSON')


