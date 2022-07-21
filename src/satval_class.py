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
import ee
import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyproj.crs import CRS
from pyproj import Transformer
from shapely.geometry import Point
# try:
#     import modin.pandas as pd
# except:
#     import pandas as pd

sys.path.append(".")
from src import satval_utils as svu


def _unique_within(all_data, gdf_pgon):
    all_data_unique = all_data.groupby(
        ['longitude', 'latitude']).size().reset_index().rename(columns={
            0: 'count'
        }).sort_values("count", ascending=False).reset_index(drop=True)

    points_unique = gpd.GeoDataFrame(geometry=[
        Point(lon, lat) for lon, lat in zip(all_data_unique.longitude.values,
                                            all_data_unique.latitude.values)
    ],
                                     index=np.arange(0, len(all_data_unique)),
                                     crs=CRS.from_epsg(4326))
    points_unique["longitude"] = all_data_unique["longitude"]
    points_unique["latitude"] = all_data_unique["latitude"]

    inside_points = gpd.sjoin(points_unique, gdf_pgon,
                              op='within').drop(columns=["geometry"])
    outside_points = all_data.merge(inside_points,
                                    on=["latitude", "longitude"],
                                    how="left")
    outside_points = outside_points[pd.isna(outside_points["AREA"])]
    outside_points_idx = np.array(
        list(set(outside_points.index.values.flatten())))

    return outside_points_idx


class satval():

    def __init__(self,
                 path_params,
                 column_map={
                     'datetime': 'datetime',
                     'longitude': 'longitude',
                     'latitude': 'latitude'
                 },
                 verbose=True):

        # Get parameters, etc. in order
        self.params = svu.parse_params(path_params)
        self.paths = svu.prepare_paths(self.params['path_base'],
                                       self.params['path_data'],
                                       self.params['path_bounding_pgon'])
        self.crs_params = svu.satellite_params(self.params['satellite'])
        self.verbose = verbose
        self.ndata = {
        }  # Dictionary to store how much data remain after various filtering, aggregating, etc.

        # Ensure the columns are parseable
        self.column_names = pd.read_csv(self.paths['data'],
                                        nrows=1).columns.tolist()
        if column_map['datetime'] not in self.column_names:
            print('Specify the correct "datetime" column.')
        if column_map['longitude'] not in self.column_names:
            print('Specify the correct "longitude" column.')
        if column_map['latitude'] not in self.column_names:
            print('Specify the correct "latitude" column.')

        if os.path.exists(self.paths['filtered']):
            self.dateloc = pd.read_csv(self.paths['filtered'],
                                       parse_dates=[column_map['datetime']])

        if os.path.exists("data/skiprows.pkl"):
            self.skiprows = pickle.load(open("data/skiprows.pkl", "rb"))

        if os.path.exists(self.paths['aggregated']):
            self.aggregated = pd.read_csv(self.paths['aggregated'])

    def get_points(
        self,
        column_map={
            'datetime': 'datetime',
            'longitude': 'longitude',
            'latitude': 'latitude'
        },
    ):
        """Reads all_data.csv and retains points that satisfy the following criteria:
            * within the CB footprint (chk_water_only.shp)
            * is a surface observation (in the top 1m)
            * has a timestamp during the MODIS overpass window time-of-day
        """
        # Get a vector of dates and locations
        if self.verbose is True:
            print('Loading in dates and locations from observations file...')
        self.dateloc = pd.read_csv(self.paths['data'],
                                   sep=',',
                                   usecols=[
                                       column_map['datetime'],
                                       column_map['longitude'],
                                       column_map['latitude']
                                   ],
                                   parse_dates=[column_map['datetime']],
                                   squeeze=True)
        self.ndata['raw'] = self.dateloc.shape[0]

        # Create an array to store the locations of rows to skip
        self.skiprows = np.zeros(len(self.dateloc), dtype=np.bool)
        self.skiprows_id = np.zeros(len(self.dateloc), dtype=np.int)

        # Determine any dates and/or locations with NaNs
        r_na, _ = np.where(self.dateloc.isna())
        self.skiprows[r_na] = True
        self.skiprows_id[r_na] = 1
        self.ndata['dateloc_nan_post'] = len(self.skiprows) - np.sum(
            self.skiprows)
        if len(r_na) > 0:
            print(
                '{} entries were removed due to nan values in datetime or lat/lon columns.'
                .format(len(r_na)))

        self.data_path = str(Path(self.paths["bounding_pgon"]).parent.parent)

        # Determine which points are outside the provided bounding polygon
        if self.verbose is True:
            print('Finding observations within the supplied polygon...')
        gdf_pgon = gpd.read_file(self.paths['bounding_pgon'])
        if gdf_pgon.crs.to_epsg() != 4326:
            gdf_pgon = gdf_pgon.to_crs(CRS.from_epsg(4326))

        outside_points_idx = _unique_within(self.dateloc, gdf_pgon)

        self.skiprows[outside_points_idx] = True
        self.skiprows_id[outside_points_idx] = 2
        self.ndata['bounding_box_post'] = len(self.skiprows) - np.sum(
            self.skiprows)

        # Determine valid rows using supplied filters
        if 'filters' in self.params.keys():
            if self.verbose is True:
                print('Applying user-supplied filters...')
            for f in self.params['filters'].keys():
                if f == 'time_of_day':
                    decimal_hours = np.array([
                        t.hour + t.minute / 60 + t.second / 3600
                        for t in self.dateloc.datetime
                    ])
                    self.skiprows[np.logical_or(
                        decimal_hours < self.params['filters'][f][0],
                        decimal_hours > self.params['filters'][f][1])] = True
                    self.skiprows_id[np.logical_or(
                        decimal_hours < self.params['filters'][f][0],
                        decimal_hours > self.params['filters'][f][1])] = 3

                else:
                    data_to_filter = pd.read_csv(self.paths['data'],
                                                 usecols=[f])
                    self.skiprows[np.logical_or(
                        data_to_filter.values < self.params['filters'][f][0],
                        data_to_filter.values >
                        self.params['filters'][f][1]).flatten()] = True
                    self.skiprows_id[np.logical_or(
                        data_to_filter.values < self.params['filters'][f][0],
                        data_to_filter.values >
                        self.params['filters'][f][1]).flatten()] = 4

        self.ndata['filter_post'] = len(self.skiprows) - np.sum(self.skiprows)

        # Check for duplicate entries here
        if self.verbose is True:
            print(
                'Removing duplicate entries and rows with nodata in time/location...'
            )

        duplicates = self.dateloc.duplicated()
        self.skiprows[duplicates.values] = True
        pickle.dump(self.skiprows, open("data/skiprows.pkl", "wb"))
        self.skiprows_id[duplicates.values] = 5
        self.ndata['duplicates'] = len(self.skiprows) - np.sum(self.skiprows)

        # Apply all filters to include only valid data
        self.dateloc = self.dateloc[~self.skiprows]

        # Add a column to track data's original row location in the full
        # database
        self.dateloc['orig_row_idx'] = np.where(self.skiprows == False)[0]

    def assign_unique_location_ids(self):
        """
        Computes a unique location id for each entry in the dataset and appends
        this to the dateloc dataframe.
        """
        if self.verbose is True:
            print('Generating unique location ids...')

        lonlats = list(
            zip(self.dateloc.longitude.values, self.dateloc.latitude.values))
        uniques = set(lonlats)
        unique_ids = np.arange(0, len(uniques))
        unique_map = {k: v for k, v in zip(uniques, unique_ids)}
        lonlat_ids = [unique_map[ll] for ll in lonlats]
        self.dateloc['loc_id'] = lonlat_ids
        self.dateloc.to_csv(self.paths['filtered'], index=False)

    def map_coordinates_to_pixels(self):
        if 'loc_id' not in self.dateloc.keys():
            raise ('Must assign unique location ids before mapping.')

        if self.verbose is True:
            print('Creating or reading shapefile of pixel centers...')

        # Create GeoDataFrame with pixel centers
        if os.path.isfile(self.paths['pixel_centers']):
            self.gdf_pix_centers = gpd.read_file(self.paths['pixel_centers'])
        else:
            self.gdf_pix_centers = svu.pixel_centers_gdf(
                self.crs_params,
                path_bounding_pgon=self.paths['bounding_pgon'],
                frac_pixel_thresh=self.params['frac_pixel_thresh'])
            self.gdf_pix_centers.to_file(self.paths['pixel_centers'])

        if self.verbose is True:
            print('Mapping observations to their nearest pixel centers...')

        # Map observation locations to pixel centers
        self.dateloc['pix_id'], self.dateloc[
            'nearest_pix_dist'], pix_ys, pix_xs = svu.map_pts_to_pixels(
                self.gdf_pix_centers, self.dateloc, self.crs_params)

        # Replace latitude and longitude of each point with its nearest pixel lat/lon
        # Must convert pixel center coordinates to 4326
        transformer = Transformer.from_crs(self.crs_params['proj4'],
                                           "EPSG:4326",
                                           always_xy=True)
        self.dateloc.longitude, self.dateloc.latitude = transformer.transform(
            pix_xs, pix_ys)

        # Threshold the mapped observations by their distance to the nearest
        # pixel. We originally mapped all the observations, regardless of their
        # distance. Here we ensure that the observation is actually within
        # the pixel's footprint.
        thresh_dist = np.sqrt(self.crs_params['gt'][0]**2 +
                              self.crs_params['gt'][4]**2) * 1.05 / 2
        self.dateloc = self.dateloc[
            self.dateloc.nearest_pix_dist <= thresh_dist]
        self.ndata['mapped_to_pixel_centers'] = self.dateloc.shape[0]

        self.dateloc.to_csv(self.paths['filtered'], index=False)

    def aggregate_data_to_unique_pixeldays(self):

        # Check that provided data columns are available
        for dc in self.params['data_cols']:
            if dc not in self.column_names:
                raise KeyError('{} is not found in the .csv'.format(dc))

        if self.verbose is True:
            print('Loading requested observations from data file...')

        # Add a 'pixelday' field to aggregate over unique pixel/day combos
        self.dateloc['pixelday'] = [
            str(pid) + '_' + str(dt.year) + '{:02d}'.format(dt.month) +
            '{:02d}'.format(dt.day) for pid, dt in zip(
                self.dateloc.pix_id.values, self.dateloc.datetime)
        ]

        # Load the desired variables to add to dateloc dataframe
        skiprowlocs = np.array(
            list(
                set(np.arange(0, len(self.skiprows))) -
                set(self.dateloc.orig_row_idx.values))) + 1
        data_df = pd.read_csv(self.paths['data'],
                              sep=',',
                              usecols=self.params['data_cols'],
                              skiprows=skiprowlocs,
                              squeeze=True)

        # Merge the data into the dateloc dataframe, but first identify the
        # rows with all nans for later removal
        data_na_rows = np.where(
            pd.isna(data_df).sum(axis=1) == len(self.params['data_cols']))[0]
        data_df['orig_row_idx'] = self.dateloc.orig_row_idx.values
        self.dateloc = self.dateloc.merge(data_df, on='orig_row_idx')

        # No longer need the data dataframe
        del data_df

        # Remove rows without valid data
        if self.verbose is True:
            print('Filtering rows whose observations are all NaN...')
        self.dateloc = self.dateloc.drop(data_na_rows, axis=0)
        self.ndata['drop_na_observation_rows'] = self.dateloc.shape[0]

        if self.verbose is True:
            print('Aggregating observations to unique pixel/days...')
        self.aggregated = svu.aggregate_to_pixeldays(self.dateloc,
                                                     self.params['data_cols'])
        self.ndata['aggregated'] = self.aggregated.shape[0]
        self.aggregated.to_csv(self.paths['aggregated'], index=False)

    def start_gee_bandval_retrieval(self, asset=None):
        """
        Starts a task on Earth Engine to grab all the pixel values from the
        imageCollection defined by dataset and specified by the locations and
        times found in self.aggregated dataframe. A minimal GeoDataFrame is 
        created for upload to GEE. 
        
        self.task is added as an attribute; this may be queried to check the
        status of the task with 'self.task.state()'.
        """
        # Begins a GEE task to export a .csv with band values from dataset
        too_big = False
        if len(
                self.aggregated
        ) < 20000 or asset is not None:  # This will certainly surpass the upload filesize allowed by the API
            try:
                if asset is None:
                    if self.verbose is True:
                        print('Making GeoDataFrame for upload to GEE...')
                    gee_gdf = svu.make_gdf_for_GEE(self.aggregated)
                else:
                    gee_gdf = None
                if self.verbose is True:
                    print('Submitting task to GEE...')
                self.task = svu.gee_fetch_bandvals(
                    gee_gdf, self.params['satellite'],
                    self.paths['gee_export_name'], asset,
                    self.params['gdrive_folder'])
                print(
                    'Task successfully started. If it completes, a file named {} will be created in your GDrive folder.'
                    .format(self.paths['gee_export_name']))
                print(
                    'You can check the status of your task with satval_obj.task.status().'
                )
            except ee.ee_exception.EEException:
                too_big = True
        else:
            gee_df = gee_gdf = svu.make_gdf_for_GEE(self.aggregated, df=True)
            too_big = True

        if too_big is True:
            gee_df.to_csv(self.paths['gee_asset_upload'], index=False)
            raise Exception(
                "The table is too large to upload via the GEE API. You must upload the csv found at {} as a GEE asset, then re-run this and supply the asset path."
                .format(self.paths['gee_asset_upload']))

    def merge_bandvals_and_data(self, path_bandvals):

        bandvals = pd.read_csv(path_bandvals)
        bandvals = bandvals.drop(columns=[
            '.geo', 'system:time_start', 'granule_pnt', 'system:index',
            'orbit_pnt'
        ])

        # Convert quality bands to interpretable values
        qbands = svu.mydocga_convert_quality_bands(bandvals)

        # Append these values to the bandvals df
        bandvals = bandvals.drop(
            columns=[k for k in bandvals.keys() if 'QC' in k])
        bandvals = pd.concat([bandvals, qbands], axis=1)

        self.aggregated = self.aggregated.merge(bandvals, on='pixelday')

        self.aggregated.to_csv(self.paths['aggregated_w_bandvals'],
                               index=False)
