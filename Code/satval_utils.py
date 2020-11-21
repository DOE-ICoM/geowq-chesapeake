# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:36:19 2020

@author: Jon
"""
import numpy as np
from shapely.geometry import  box, Point, Polygon, MultiPolygon, GeometryCollection
from pyproj import CRS
import geopandas as gpd
import pyproj
from scipy.spatial import cKDTree
import pandas as pd
import geojson
import ee


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


def pixel_centers_gdf(crs_params, path_bounding_pgon=None, frac_pixel_thresh=None):
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
    frac_pixel_thresh : float, optional
        The minimum fraction of each pixel's footprint that is within the
        bounding polygon. For example, a value of 1 indicates that a pixel
        must fully be within the bounding polygon to be kept. A value of 0.5
        indicates that at least half of the pixel's area must be within the
        bounding polygon to be kept. If this parameter is specified, 
        path_bounding_pgon must also be specified.

    Returns
    -------
    pix_centers : GeoPandas GeoDataFrame
        Stores the geometry of each pixel center as well as their IDs. Note that
        IDs are not random, but are the index of the pixel within the image
        as found via np.unravel_index().
    """
    # Check inputs
    if frac_pixel_thresh is not None and path_bounding_pgon is None:
        raise ValueError('Must specify path_bounding_pgon if specifying frac_pixel_thresh.')
        
    
    if path_bounding_pgon is not None:
        bb_gdf = gpd.read_file(path_bounding_pgon)
        bb_gdf = bb_gdf.to_crs(CRS.from_proj4(crs_params['proj4']))
        bb = bb_gdf.geometry.values[0].bounds
        col_min = np.floor((bb[0] - crs_params['gt'][2]) / crs_params['gt'][0])
        col_max = np.ceil((bb[2] - crs_params['gt'][2]) / crs_params['gt'][0])
        row_max = np.floor((bb[1] - crs_params['gt'][5]) / crs_params['gt'][4])
        row_min = np.ceil((bb[3] - crs_params['gt'][5]) / crs_params['gt'][4])
        rows = np.arange(row_min, row_max + 1, 1, dtype=np.int)
        cols = np.arange(col_min, col_max + 1, 1, dtype=np.int)
        r, c = np.meshgrid(rows, cols)
        r, c = r.flatten(), c.flatten()
        pix_idx = np.ravel_multi_index((r,c), crs_params['shape'][::-1])
        
    else: # You're gonna have a bad time
        # Get pixel centers and ids
        pix_idx = np.arange(0, np.product(crs_params['shape']))
        r, c = np.unravel_index(pix_idx, crs_params['shape'][::-1])
    
    lons = crs_params['gt'][2] + crs_params['gt'][0] * c + crs_params['gt'][0]/2
    lats = crs_params['gt'][5] + crs_params['gt'][4] * r + crs_params['gt'][4]/2
    
    # Filter pixel centers to those within the supplied polygon
    if path_bounding_pgon is not None:
        gdf_coords = gpd.GeoDataFrame(geometry=[Point(lo, la) for lo, la in zip(lons, lats)], crs=bb_gdf.crs, data={'pix_idx':pix_idx})
        intersected = gpd.sjoin(gdf_coords, bb_gdf, how='inner', op='within')
        pix_idx = intersected.pix_idx.values
        lats = []
        lons = []
        for g in intersected.geometry:
            lons.append(g.coords.xy[0][0])
            lats.append(g.coords.xy[1][0])
        lats = np.array(lats)
        lons = np.array(lons)

    # If an inclusion threshold is specified, filter for it
    if frac_pixel_thresh is not None:
        # First, find all the pixagons that are completely in the polygon
        pgons = []
        for lo, la in zip(lons, lats):
            tl = (lo - crs_params['gt'][0]/2, la - crs_params['gt'][4]/2)
            tr = (lo + crs_params['gt'][0]/2, la - crs_params['gt'][4]/2)
            bl = (lo - crs_params['gt'][0]/2, la + crs_params['gt'][4]/2)
            br = (lo + crs_params['gt'][0]/2, la + crs_params['gt'][4]/2)
            pgons.append(Polygon((tl, tr, br, bl, tl)))
        gdf_pgons = gpd.GeoDataFrame(geometry=pgons, crs=bb_gdf.crs, data={'pix_idx':pix_idx})
        intersected = gpd.sjoin(gdf_pgons, bb_gdf, how='inner', op='within')
        
        # Then check the area overlap for the pixagons that aren't completely
        # contained
        some_overlap_pids = set(gdf_pgons.pix_idx.values) - set(intersected.pix_idx.values)
        gdf_overlap = gdf_pgons.loc[gdf_pgons['pix_idx'].isin(some_overlap_pids)]
        # This is a trick to divide a large, complex bounding polygon into
        # a bunch of smaller ones for more efficient intersection
        geom_parts = katana(bb_gdf.geometry.values[0], crs_params['gt'][0]*5)
        geom_parts_gdf = gpd.GeoDataFrame(geometry=geom_parts, crs=bb_gdf.crs)
        ints = gpd.overlay(geom_parts_gdf, gdf_overlap, how="intersection")
        ints['areas'] = ints.area
        overlap_areas = ints.groupby(by='pix_idx').sum().reset_index()
        overlap_areas['overlap_frac'] = [oa/gdf_overlap.geometry.values[gdf_overlap.pix_idx.values==pi].area[0] for oa, pi in zip(overlap_areas.areas.values, overlap_areas.pix_idx.values)]
        overlap_areas = overlap_areas[overlap_areas.overlap_frac.values>=frac_pixel_thresh]
        
        # Combine the fully-in pixagons with those that have a high-enough
        # fraction of area in
        pix_ids_keep = np.append(overlap_areas.pix_idx.values, intersected.pix_idx.values)
        idcs_keep = np.array([np.where(pid==pix_idx)[0][0] for pid in pix_ids_keep])
        
        # Remove all the pixels that don't meet the threshold
        pix_idx = pix_idx[idcs_keep]
        lons = lons[idcs_keep]
        lats = lats[idcs_keep]
        
    # Export the modis pixel centers
    pix_centers = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)], crs=CRS.from_proj4(crs_params['proj4']))
    pix_centers['pix_idx'] = pix_idx
    
    return pix_centers


def katana(geometry, threshold, count=0):
    """Split a Polygon into two parts across it's shortest dimension.
        This function was pasted from 
        https://snorfalorpagus.net/blog/2016/03/13/splitting-large-polygons-for-faster-intersections/
    """
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
        b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
        b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon, MultiPolygon)):
                result.extend(katana(e, threshold, count+1))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result


def map_pts_to_pixels(pix_centers, coords_df, crs_params):
    """
    Maps each provided coordinate to its nearest pixel center as provided
    by pix_centers. Assumes coordinates have CRS EPSG:4326.

    Parameters
    ----------
    pix_centers : GeoPandas GeoDataFrame
        Contains the geometries of the pixel centers as Points, as well as 
        their IDs.
    coords_df : Pandas DataFrame
        Contains the coordinates (latitude, longitude) of the points to
        map to pixels. Assumes ESPG:4326.
    crs_params : dict
        Contains the keys: 'proj4', 'gt', and 'shape' defining the CRS of the
        image or ImageCollection.

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
    c_lons, c_lats = p(coords_df.longitude.values, coords_df.latitude.values)  
    coords_proj = [(lon, lat) for lon, lat in zip(c_lons, c_lats)]

    # Build a kd-tree of the pixel centers
    pc_lat = [g.coords.xy[1][0] for g in pix_centers.geometry.values]
    pc_lon =  [g.coords.xy[0][0] for g in pix_centers.geometry.values]
    pix_idx = pix_centers.pix_idx.values
    ckd_tree = cKDTree(list(zip(pc_lon, pc_lat)))
    
    # Find the index of the nearest pixel center to each coordinate
    dists, nearest_idx = ckd_tree.query(coords_proj, k=1)
    nearest_pix_ids = pix_idx[nearest_idx]
        
    return nearest_pix_ids, dists
    

def aggregate_to_pixeldays(df, datacols):
    """
    Averages all the observations for each unique date/location.

    Parameters
    ----------
    df : pandas DataFrame
        Contains the observations and a field over which to aggregate.

    Returns
    -------
    None.

    """
    
    def res_agg(x, args):
        
        datacols = args
            
        def nanmean_wrapper(a):
            """
            Returns np.nan if all the values are nan, else returns the mean.
            """
            if np.isnan(a).all():
                return np.nan
            else:
                return np.nanmean(a)
    
        # Define aggregation functions: for columns NOT in datacols, we just
        # take the first value of the group. For those in datacols, we 
        # compute a mean and a count. 
        # See https://stackoverflow.com/questions/20659523/issue-calling-lambda-stored-in-dictionary
        # for the lambda construction (i.e. restatement of inputs).        
        aggs = {}
        for k in x.keys():
            # if k not in datacols:
            aggs[k] = lambda x=x, k=k: x[k].values[0]
        for k in datacols:
            aggs[k] = lambda x=x, k=k: nanmean_wrapper(x[k])
            aggs[k + ' count'] =  lambda x=x, k=k: np.sum(np.isnan(x[k])==0)
                            
        do_agg = {k : aggs[k](x) for k in aggs.keys()}
    
        return pd.Series(do_agg, index=list(aggs.keys())) # This is only guaranteed in Python 3.6+ because dictionaries are ordered therein
    
    grouped = df.groupby(by='pixelday').apply(res_agg, args=datacols) 
    # grouped = grouped.reset_index()

    return grouped


def gdf_to_FC(gdf):
    """
    Converts a geopandas GeoDataFrame to an Earth Engine FeatureCollection.
    Code adapted from: https://github.com/gee-community/eeconvert/blob/master/eeconvert/__init__.py
    Note that this does not upload the FeatureCollection to the server, but
    only creates its client-side objects.    

    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        Must contain point geometries; other types not supported.

    Returns
    -------
    fc : ee.FeatureCollection
        An Earth Enging FeatureCollection object.
    """

    def shapelyToEEFeature(row):
        properties = row.drop(["geometry"]).to_dict()
        geoJSONfeature = geojson.Feature(geometry=row["geometry"], properties=properties)
        return ee.Feature(geoJSONfeature)

    gdfCopy = gdf.copy()
    gdfCopy["eeFeature"] = gdfCopy.apply(shapelyToEEFeature,1)
    featureList = gdfCopy["eeFeature"].tolist()
    fc =  ee.FeatureCollection(featureList)
    
    return fc


def gee_fetch_bandvals(gdf, dataset, filename, gdrive_folder=None):
    """
    Starts a task on Google Earth Engine to fetch all the band values corresponding
    to the locations and times provided in gdf. If the task is successful,
    exports a .csv with band values from dataset to your GDrive.

    Parameters
    ----------
    gdf : geopandas GeoDataFrame
        Contains Point geometries for all the locations for which to pull band
        values. Also must contain 'system:time_start' field that contains the 
        nanseconds since Unix epoch at which the observation was taken. Should
        also contain at least one unique ID field that can be used to map
        the exported .csv back to the one containing the data values.
    dataset : str
        The ID of the imageCollection on GEE to pull. Currently supports
        'MYD09GA', 'MYD09GQ', and 'MYDOCGA'.
    filename : str
        The name of the exported .csv without extension.
    gdrive_folder : str, optional
        If provided, GEE will export the .csv to a folder with this name on
        your GDrive. If this folder doesn't exit, GEE will create it.
        

    Returns
    -------
    bandValFC : TYPE
        DESCRIPTION.

    """
    
    ee.Initialize()
    
    def getBandValues(im):
        # Filter the observations to the date range of the image
        obsSameDate = obs.filterDate(ee.Date(im.get('system:time_start')), ee.Date(im.get('system:time_end')))
    
        bandValFC = ee.Algorithms.If(
            obsSameDate.size(), # If this is 0, we have no images in the collection and it evaluates to "False", else "True"
            im.reduceRegions(obsSameDate, ee.Reducer.first(), scale=500), #  True: grab its band values
            None # False: only geometry and 'pixelday' properties will be set
        )
    
        return bandValFC
    
    # We need to convert the GeoDataFrame to a server-side FeatureCollection
    # This will be uploaded when the task is started.
    obs = gdf_to_FC(gdf)

    # Define some modis asset locations on GEE
    icAssets = {'MYD09GA' : "MODIS/006/MYD09GA",
              'MYD09GQ' : 'MODIS/006/MYD09GQ',
              'MYDOCGA' : 'MODIS/006/MYDOCGA'}
            
    # Load the imageCollection, get date range
    ic = ee.ImageCollection(icAssets[dataset])
    icDateRange = ic.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
    
    # Filter the observations to the imageCollection dateRange
    obs = obs.filterDate(ee.Date(icDateRange.get('min')), ee.Date(icDateRange.get('max')))
    
    # ic = ee.ImageCollection(ic.toList(10, 1000))
    bandVals = ic.map(getBandValues, opt_dropNulls=True).flatten()
          
    # Export the dataframe
    if gdrive_folder is None:
        task = ee.batch.Export.table.toDrive(
          collection = bandVals,
          description = filename,
          fileFormat = 'CSV',
        )
    else:
        task = ee.batch.Export.table.toDrive(
          collection = bandVals,
          description = filename,
          fileFormat = 'CSV',
          folder = gdrive_folder
        )

    
    task.start()
    
    return task
