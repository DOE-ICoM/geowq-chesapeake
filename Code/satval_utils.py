# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:36:19 2020

@author: Jon
"""
import numpy as np
from shapely.geometry import Point
from pyproj import CRS
import geopandas as gpd
import pyproj
from scipy.spatial import cKDTree


def satellite_params(dataset):
    """
    Returns parameters for a given remotely sensed dataset found in GEE.
    Currently supports: MYDOCGA.006

    Parameters
    ----------
    dataset : str
        The dataset name as given by Google Earth Engine Data Catalog:
            https://developers.google.com/earth-engine/datasets

    Returns
    -------
    None.

    """
    ds_params = {}
    if dataset == 'MYDOCGA.006':
        ds_params['proj4'] = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs' 
        ds_params['gt'] = (926.625433056, 0, -20015109.354, 0, -926.625433055, 10007554.677) # geotransform pulled from GEE: (xScale, xShearing, xTranslation, yShearing, yScale, yTranslation)
        ds_params['shape'] = (43200,21600) # ncols, nrows

    return ds_params


def pixel_centers_gdf(crs_params, path_bounding_pgon=None):
    """
    Makes a GeoDataFrame containing the centers of each pixel in the image
    defined by crs_params. A bounding polygon can be supplied to greatly 
    reduce the number of pixel centers; this drastically saves time for the
    following steps of mapping each observation to its nearest pixel.

    Parameters
    ----------
    crs_params : dict
        Contains the keys: 'proj4', 'gt', and 'shape' defining the CRS of the
        image or ImageCollection.
    path_bounding_pgon : str, optional
        Path to a bounding polygon shapefile/geojson.

    Returns
    -------
    pix_centers : GeoPandas GeoDataFrame
        Stores the geometry of each pixel center as well as their IDs. Note that
        IDs are not random, but are the index of the pixel within the image
        as found via np.unravel_index().
    """
    # Get pixel centers and ids
    pix_idx = np.arange(0, np.product(crs_params['shape']))
    r, c = np.unravel_index(pix_idx, crs_params['shape'][::-1])
    lons = crs_params['gt'][2] + crs_params['gt'][0] * c + crs_params['gt'][0]/2
    lats = crs_params['gt'][5] + crs_params['gt'][4] * r + crs_params['gt'][4]/2
    
    # Filter points to those within our bounding box
    if path_bounding_pgon is not None:
        bb_gdf = gpd.read_file(path_bounding_pgon)
        bb_gdf = bb_gdf.to_crs(CRS.from_proj4(crs_params['proj4']))
        extents = bb_gdf.geometry[0].bounds
        out = np.logical_or(lons < extents[0], np.logical_or(lons > extents[2], np.logical_or(lats > extents[3], lats < extents[1])))
        pix_idx = pix_idx[~out]
        lons = lons[~out]
        lats = lats[~out]
    
    # Export the modis pixel centers
    pix_centers = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)], crs=CRS.from_proj4(crs_params['proj4']))
    pix_centers['pix_idx'] = pix_idx
    
    return pix_centers


def map_pts_to_pixels(pix_centers, coords, crs_params):
    """
    Maps each provided coordinate to its nearest pixel center as provided
    by pix_centers. Assumes coordinates have CRS EPSG:4326.

    Parameters
    ----------
    pix_centers : GeoPandas GeoDataFrame
        Contains the geometries of the pixel centers as Points, as well as 
        their IDs.
    coords : Pandas DataFrame
        Contains the coordinates (latitude, longitude) of the points to
        map to pixels. Assumes ESPG:4326.

    Returns
    -------
    pix_ids : np.array()
        A vector the same length as coords that contains each point's nearest
        pixel ID. 
    """
    
    # pix_centers = cd.gdf_pix_centers
    # coords = cd.dateloc
    
    # Reproject the coordinates to match the pixel centers
    p = pyproj.Proj(crs_params['proj4'])
    c_lons, c_lats = p(coords.longitude.values, coords.latitude.values)  
    coords_proj = [(lon, lat) for lon, lat in zip(c_lons, c_lats)]

    # Build a kd-tree of the pixel centers
    pc_lat = [g.coords.xy[1][0] for g in pix_centers.geometry.values]
    pc_lon =  [g.coords.xy[0][0] for g in pix_centers.geometry.values]
    pix_idx = pix_centers.pix_idx.values
    
    # Build a kd-tree of the points (this can take awhile)
    ckd_tree = cKDTree(list(zip(pc_lon, pc_lat)))
    
    # Find the index of the nearest pixel center to each coordinate
    dist, idx = ckd_tree.query(coords_proj, k=1)
    pix_ids = pix_idx[idx]
        
    return pix_ids
    
