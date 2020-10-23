# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:08:47 2020

@author: Jon
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pyproj import CRS

path_dummy = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Buoys\all_buoys.csv" # path to full observation dataset
path_out = r'' # path to save dataset with location and pixel ids appended
ds = pd.read_csv(path_dummy)

# Give each location a unique lat/lon
latlons = list(zip(ds.longitude.values, ds.latitude.values))
uniques = set(latlons)
unique_ids = np.arange(0,len(uniques))
unique_map = {k:v for k, v in zip(uniques, unique_ids)}
ll_ids = [unique_map[ll] for ll in latlons]
ds['loc_id'] = ll_ids
# GeoDataframe with map
u_keys = list(unique_map.keys())
u_vals = list(unique_map.values())
geoms = [Point(uk) for uk in u_keys]
unique_df = gpd.GeoDataFrame(geometry=geoms, crs=CRS.from_epsg(4326))
unique_df['loc_id'] = u_vals

# Map the locations to their proper pixels
# We only need to map the unique locations

# Some MODIS grid info for MYDOCGA
modis_proj4 = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs' # pulled from GEE
modis_gt = (926.625433056, 0, -20015109.354, 0, -926.625433055, 10007554.677) # geotransform pulled from GEE: (xScale, xShearing, xTranslation, yShearing, yScale, yTranslation)
modis_shape = (43200,21600) # ncols, nrows

# We need to reproject the lat/lon coordinates to match the projection of the MODIS grid
unique_df = unique_df.to_crs(CRS.from_proj4(modis_proj4))

# Construct the grid of pixel coordinates
left = modis_gt[2] + modis_gt[0] * np.arange(0, modis_shape[0])
right = left + modis_gt[0]
top = modis_gt[5] + modis_gt[4] * np.arange(0, modis_shape[1])
bottom = top + modis_gt[4]

# Assign each unique location a unique pixel ID
pixel_ids = []
for g in unique_df.geometry.values:
    lonlat = g.coords.xy
    rowidx = np.where(np.logical_and(lonlat[1][0] > bottom, lonlat[1][0] < top))[0][0]
    colidx = np.where(np.logical_and(lonlat[0][0] > left, lonlat[0][0] < right))[0][0]
    pix_idx = np.ravel_multi_index((rowidx, colidx), modis_shape[::-1])
    pixel_ids.append(pix_idx)
    
# Map the pixel IDs back to the original dataframe
ds['pix_id'] = np.empty(len(ds))
for pid, lid in zip(pixel_ids, unique_df.loc_id.values):
    ds['pix_id'][ds['loc_id']==lid] = int(pid)
    
# Save the dataframe
ds.to_file(path_out)

