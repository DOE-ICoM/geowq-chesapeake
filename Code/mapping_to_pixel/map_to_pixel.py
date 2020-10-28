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
from scipy.spatial import cKDTree

path_alldata = r"C:\Users\Jon\Desktop\Research\ICoM\Data\VECOS Dataflow\VECOS_Dataflow.csv" # path to full observation dataset
path_bounding_poly = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\Boundaries\delaware_chesapeake.shp"
path_out = r'C:\Users\Jon\Desktop\Research\ICoM\Data\test_out.csv' # path to save dataset with location and pixel ids appended
ds = pd.read_csv(path_alldata)

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

# Get pixel centers and ids
p_idx = np.arange(0, np.product(modis_shape))
r, c = np.unravel_index(p_idx, modis_shape[::-1])
lons = modis_gt[2] + modis_gt[0] * c + modis_gt[0]/2
lats = modis_gt[5] + modis_gt[4] * r + modis_gt[4]/2

# Filter points to those within our bounding box
bb_gdf = gpd.read_file(path_bounding_poly)
bb_gdf = bb_gdf.to_crs(CRS.from_proj4(modis_proj4))
extents = bb_gdf.geometry[0].bounds
out = np.logical_or(lons < extents[0], np.logical_or(lons > extents[2], np.logical_or(lats > extents[3], lats < extents[1])))
p_idx = p_idx[~out]
lons = lons[~out]
lats = lats[~out]

# Build a kd-tree of the points (this can take awhile)
ckd_tree = cKDTree(list(zip(lons, lats)))

# Find the index of the nearest pixel center to each point in uniques
u_pts = [(g.coords.xy[0][0], g.coords.xy[1][0]) for g in unique_df.geometry]
dist, idx = ckd_tree.query(u_pts, k=1)
pix_ids = p_idx[idx]
    
# Map the pixel IDs back to the original dataframe
# Make a lid:pid map
loc_to_pix = {lid:pid for lid, pid in zip(unique_df.loc_id.values,pix_ids)}
ds_pix_locs = [int(loc_to_pix[lid]) for lid in ds.loc_id.values]
ds['pix_id'] = ds_pix_locs

# Save the dataframe
ds.to_csv(path_out, index=False)

# # Evaluate code
# import random
# rands = np.array(list(set([random.randrange(0, len(ds)) for i in range(10000)])))
# ds_subset = ds.iloc[rands]
# ds_subset.to_csv(r'C:\Users\Jon\Desktop\Research\ICoM\Data\test_subset.csv', index=False)

# pix_centers = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)], crs=CRS.from_proj4(modis_proj4))
# pix_centers = pix_centers.to_crs(CRS.from_epsg(4326))
# pix_centers.to_file(r'C:\Users\Jon\Desktop\Research\ICoM\Data\modis_pixel_centers.shp')
