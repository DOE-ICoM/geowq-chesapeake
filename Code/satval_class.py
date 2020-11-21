# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:42:16 2020

@author: Jon

A class for fetching band values from remotely-sensed image collections given
a csv of observations that include latitude, longitude.

Note that modin is used for parallelization of pandas tasks. Testing shows that
this greatly reduces read times. If modin doesn't have a pandas functionality 
implemented yet, it falls back to pandas. Update: modin parallelizes groupby.apply()
functions by column (rather than by group), so it cannot be used for this as
we apply different aggregations to different columns.
"""
import numpy as np
import satval_utils as svu
import os
import geopandas as gpd
from shapely.geometry import Point
from pyproj.crs import CRS
# try:
#     import modin.pandas as pd
# except:
#     import pandas as pd
import pandas as pd


class satval():
    
    def __init__(self, paths, filters=None, 
                 column_map={'datetime':'datetime', 'longitude':'longitude', 'latitude':'latitude'},
                 verbose=True):    
        
        # Ensure the required paths are set
        err = False
        for k in ['data', 'bounding_pgon', 'pixel_centers', 'aggregated']:
            if k not in paths.keys():
                print('{} path must be specified in supplied paths.'.format(k))
                err = True
        if err is True:
            raise KeyError()
        
        self.paths = paths
        self.verbose = verbose
        
        # Ensure the columns are parseable
        self.column_names = pd.read_csv(self.paths['data'], nrows=1).columns.tolist()
        if column_map['datetime'] not in self.column_names:
            print('Specify the correct "datetime" column.')
        if column_map['longitude'] not in self.column_names:
            print('Specify the correct "longitude" column.')
        if column_map['latitude'] not in self.column_names:
            print('Specify the correct "latitude" column.')
                        
        # Get a vector of dates and locations
        if self.verbose is True:
            print('Loading in dates and locations from observations file...')
        self.dateloc = pd.read_csv(self.paths['data'], sep=',', usecols=[column_map['datetime'],column_map['longitude'], column_map['latitude']], 
                            parse_dates = [column_map['datetime']], squeeze=True)

        # Create an array to store the locations of rows to skip
        self.skiprows = np.zeros(len(self.dateloc), dtype=np.bool)
        
        # Determine any dates and/or locations with NaNs
        n_pre = self.dateloc.shape[0]
        self.skiprows[pd.isna(self.dateloc).sum(axis=1)] = True
        n_post = self.dateloc.shape[0]
        if n_pre - n_post > 0:
            print('{} entries were removed due to nan values.'.format(n_pre-n_post))
 
        if self.verbose is True:
            print('Finding observations within the supplied polygon...')
        # Determine which points are outside the bounds of the provided bounding polygon
        gdf_pgon = gpd.read_file(self.paths['bounding_pgon'])
        if gdf_pgon.crs.to_epsg() != 4326:
            gdf_pgon = gdf_pgon.to_crs(CRS.from_epsg(4326))
        points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(self.dateloc.longitude.values, self.dateloc.latitude.values)],
                                  index=np.arange(0, len(self.dateloc)),
                                  crs=CRS.from_epsg(4326))
        inside_points = gpd.sjoin(points, gdf_pgon, op='within')
        outside_points_idx = np.array(list(set(points.index.values.flatten()) - set(inside_points.index.values.flatten())))
        self.skiprows[outside_points_idx] = True
    
        # Determine valid rows using supplied filters
        if filters is not None:
            for f in filters.keys():
                if f == 'time_of_day':
                    decimal_hours = np.array([t.hour + t.minute/60 + t.second/3600 for t in self.dateloc.datetime])
                    self.skiprows[np.logical_and(decimal_hours>filters[f][0], decimal_hours<=filters[f][1])] = True
                else:
                    data_to_filter = pd.read_csv(self.paths['data'], usecols = [f])
                    self.skiprows[np.logical_and(data_to_filter.values>filters[f][0], data_to_filter.values<=filters[f][1]).flatten()] = True
                           
                    
        # Check for duplicate entries here
        if self.verbose is True:
            print('Removing duplicate entries and rows with nodata in time/location...')

        duplicates = self.dateloc.duplicated()
        self.skiprows[duplicates.values] = True
                    
        # Apply filters
        self.dateloc = self.dateloc[~self.skiprows]
        
        # Compute GEE times
        self.dateloc['system:time_start'] = (pd.to_datetime(self.dateloc['datetime'].values).astype(np.int64) / int(1e6)).astype(np.int64)
        
                                  
    def assign_unique_location_ids(self):
        """
        Computes a unique location id for each entry in the dataset and appends
        this to the dateloc dataframe.
        """
        if self.verbose is True:
            print('Generating unique location ids...')

        lonlats = list(zip(self.dateloc.longitude.values, self.dateloc.latitude.values))
        uniques = set(lonlats)
        unique_ids = np.arange(0, len(uniques))
        unique_map = {k:v for k, v in zip(uniques, unique_ids)}
        lonlat_ids = [unique_map[ll] for ll in lonlats]
        self.dateloc['loc_id'] = lonlat_ids
        

    def map_coordinates_to_pixels(self, dataset, frac_pixel_thresh=None):
        if 'loc_id' not in self.dateloc.keys():
            raise('Must assign unique location ids before mapping.')

        self.crs_params = svu.satellite_params(dataset)
        
        if self.verbose is True:
            print('Creating or reading shapefile of pixel centers...')
            
        # Create GeoDataFrame with pixel centers
        if os.path.isfile(self.paths['pixel_centers']) is True:
            self.gdf_pix_centers = gpd.read_file(self.paths['pixel_centers'])
        else:
            self.gdf_pix_centers = svu.pixel_centers_gdf(self.crs_params, path_bounding_pgon=self.paths['bounding_pgon'], frac_pixel_thresh=frac_pixel_thresh)
            self.gdf_pix_centers.to_file(self.paths['pixel_centers'])

        if self.verbose is True:
            print('Mapping observations to their nearest pixel centers...')

        # Map observation locations to pixel centers
        self.dateloc['pix_id'], self.dateloc['nearest_pix_dist'] = svu.map_pts_to_pixels(self.gdf_pix_centers, self.dateloc, self.crs_params)
        
    
    def aggregate_data_to_unique_pixeldays(self, datacols):
        
        # Check that provided data columns are available
        for dc in datacols:
            if dc not in self.column_names:
                raise KeyError('{} is not found in the .csv'.format(dc))

        if self.verbose is True:
            print('Loading requested observations from data file...')

        # Add a 'pixelday' field to aggregate over unique pixel/day combos
        self.dateloc['pixelday'] = [str(pid) + '_' + str(dt.year) + '{:02d}'.format(dt.month) + '{:02d}'.format(dt.day) for pid, dt in zip(self.dateloc.pix_id.values, self.dateloc.datetime)]
        
        # Load the desired variables and add to dateloc dataframe       
        skiprowlocs = np.where(self.skiprows==True)[0] + 1 # Have to add one to account for header
        data_df = pd.read_csv(self.paths['data'], sep=',', usecols=datacols,
                    skiprows=skiprowlocs, squeeze=True)
        # Merge the dataframes
        for dc in datacols:
            self.dateloc[dc] = data_df[dc]
        del data_df
        
        if self.verbose is True:
            print('Filtering rows whose observations are all NaN...')

        # Filter out observations that are too far away from a pixel
        thresh_dist = np.sqrt(self.crs_params['gt'][0]**2 + self.crs_params['gt'][4]**2)*1.01
        self.dateloc = self.dateloc[self.dateloc.nearest_pix_dist<=thresh_dist]
        
        # Remove rows whose data columns are all nans
        self.dateloc = self.dateloc.dropna(subset = datacols, how='all')
        
        if self.verbose is True:
            print('Aggregating observations to unique pixel/days...')

        self.aggregated = svu.aggregate_to_pixeldays(self.dateloc, datacols)


    def start_gee_bandval_retrieval(self, dataset, filename, gdrive_folder=None):
        """
        Starts a task on Earth Engine to grab all the pixel values from the
        imageCollection defined by dataset and specified by the locations and
        times found in self.aggregated dataframe. A minimal GeoDataFrame is 
        created for upload to GEE. 
        
        self.task is added as an attribute; this may be queried to check the
        status of the task with 'self.task.state'.
        """
        if self.verbose is True:
            print('Making GeoDataFrame for upload to GEE...')

        # Create a geodataframe for conversion to EE FeatureCollection
        # Only need the geometry, system:time_start, and pixelday
        gee_gdf = gpd.GeoDataFrame(geometry=[Point(lo, la) for lo, la in zip(self.aggregated.longitude.values, self.aggregated.latitude.values)], 
                                   crs=CRS.from_proj4(self.crs_params['proj4']))
        copy_props = ['system:time_start', 'pixelday']
        for prop in copy_props:
            gee_gdf[prop] = self.aggregated[prop].values
        
        if self.verbose is True:
            print('Uploading table and submitting task to GEE...')

        # Begins a GEE task to export a .csv with band values from dataset
        self.task = svu.gee_fetch_bandvals(gee_gdf, dataset, filename, gdrive_folder)
        