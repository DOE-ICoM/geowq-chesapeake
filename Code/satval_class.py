# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:42:16 2020

@author: Jon

A class for fetching band values from remotely-sensed image collections given
a csv of observations that include latitude, longitude.

Note that modin is used for parallelization of pandas tasks. Testing shows that
this greatly reduces read times. If modin doesn't have a pandas functionality 
implemented yet, it falls back to pandas.
"""
import numpy as np
import satval_utils as svu
import os
import geopandas as gpd
try:
    import modin.pandas as pd
except:
    import pandas as pd


class satval():
    
    def __init__(self, paths, filters=None, 
                 column_map={'datetime':'datetime', 'longitude':'longitude', 'latitude':'latitude'}):    
        
        # Ensure the required paths are set
        err = False
        for k in ['data', 'bounding_pgon', 'pixel_centers', 'aggregated']:
            if k not in paths.keys():
                print('{} path must be specified in supplied paths.'.format(k))
                err = True
        if err is True:
            raise KeyError()
        
        self.paths = paths
        
        # Ensure the columns are parseable
        self.column_names = pd.read_csv(path_data, nrows=1).columns.tolist()
        if column_map['datetime'] not in self.column_names:
            print('Specify the correct "datetime" column.')
        if column_map['longitude'] not in self.column_names:
            print('Specify the correct "longitude" column.')
        if column_map['latitude'] not in self.column_names:
            print('Specify the correct "latitude" column.')
            
        # Determine valid rows based on provided filters
        if filters is not None:
            for f in filters.keys():
                break
            
        # Get a vector of dates and locations
        self.dateloc = pd.read_csv(path_data, sep=',', usecols=[column_map['datetime'],column_map['longitude'], column_map['latitude']], 
                            parse_dates = [column_map['datetime']], squeeze=True)
                        
        # Remove any dates and/or locations with NaNs
        n_pre = self.dateloc.shape[0]
        self.dateloc.dropna(axis=0, inplace=True)
        n_post = self.dateloc.shape[0]
        if n_pre - n_post > 0:
            print('{} entries were removed due to nan values.'.format(n_pre-n_post))
            
    
    def filter_by(self, hours=None, depth=None):
        
        if hours is not None:
            decimal_hours = self.dateloc.datetime
        
        
                                  
    def assign_unique_location_ids(self):
        """
        Computes a unique location id for each entry in the dataset and appends
        this to the dateloc dataframe.
        """
        
        lonlats = list(zip(self.dateloc.longitude.values, self.dateloc.latitude.values))
        uniques = set(lonlats)
        unique_ids = np.arange(0, len(uniques))
        unique_map = {k:v for k, v in zip(uniques, unique_ids)}
        lonlat_ids = [unique_map[ll] for ll in lonlats]
        self.dateloc['loc_id'] = lonlat_ids
        

    def map_locations_to_pixels(self, dataset='MYDOCGA.006'):
        if 'loc_id' not in self.dateloc.keys():
            raise('Must assign unique location ids before mapping.')

        self.crs_params = svu.satellite_params(dataset)
        
        # Create GeoDataFrame with pixel centers
        if os.path.isfile(self.paths['pixel_centers']) is True:
            self.gdf_pix_centers = gpd.read_file(self.paths['pixel_centers'])
        else:
            self.gdf_pix_centers = svu.pixel_centers_gdf(self.crs_params, path_bounding_pgon=self.paths['bounding_pgon'])
            self.gdf_pix_centers.to_file(self.paths['pixel_centers'])

        # # Map observations to pixel centers
        self.dateloc['pix_id'] = svu.map_pts_to_pixels(self.gdf_pix_centers, self.dateloc, self.crs_params)
        
    
    # def aggregate_to_unique_pixeldays(self):
        

        

# ds['loc_id'] = ll_ids
# # GeoDataframe with map
# u_keys = list(unique_map.keys())
# u_vals = list(unique_map.values())
# geoms = [Point(uk) for uk in u_keys]
# unique_df = gpd.GeoDataFrame(geometry=geoms, crs=CRS.from_epsg(4326))
# unique_df['loc_id'] = u_vals
