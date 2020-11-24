# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:53:35 2020

@author: Jon

env: icomdata
"""
import sys
sys.path.append(r'C:\Users\Jon\Desktop\Research\ICoM\satval\Code')
from satval_class import satval

""" The following block represents inputs that MUST be defined before running. """
path_storage = r'C:\Users\Jon\Desktop\Research\ICoM\Data\Processed Data'
path_data = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data.csv"
path_bounding_pgon = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\Boundaries\chk_water_only.shp"

# Filtering parameters    
hours_filt = [17.1, 19.3] # which hours of the day to filter the observations to
depth_filt = [0, 1] # what is the deepest measurement to keep
data_columns = ['SST (C)', 'turbidity (NTU)', 'SSS (psu)', 'depth (m)'] # which data should be kept and aggregated
frac_pixel_thresh = .9 # what fraction of each pixel's footprint must be inside the bounding polygon to include it
filters = {'depth (m)': depth_filt, 'time_of_day' : hours_filt}

# Google Earth Engine parameters
satellite_data = 'MYDOCGA.006' # GEE ID of dataset we want pixel values from
gdrive_folder = 'ICOM exports' # where to export GEE task results (folder in your GDrive, will be created if doesn't exist)


""" Processing begins here. """
cd = satval(path_data, path_storage, path_bounding_pgon, satellite_data, filters=filters)
cd.assign_unique_location_ids()
cd.map_coordinates_to_pixels(frac_pixel_thresh=frac_pixel_thresh)
cd.aggregate_data_to_unique_pixeldays(datacols=data_columns)

# Try to start a GEE task by directly uploading the geodataframe - this 
# will fail for 
cd.start_gee_bandval_retrieval(gdrive_folder=gdrive_folder)
# Failed, so we upload the shapefile manually and pass in the asset location
gee_asset = 'users/jonschwenk/chk_uniques'
cd.start_gee_bandval_retrieval(asset=gee_asset, gdrive_folder=gdrive_folder)


"""Need to wait for GEE task to finish and download the .csv"""
paths['gee_bandvals'] = r"C:\Users\Jon\Desktop\Research\ICoM\Data\chk_del_full_bandvals.csv"
cd.merge_bandvals_and_data(path_bandvals=paths['gee_bandvals'], path_data=paths['aggregated'], data_columns)



""" Debugging """
import numpy as np
import pandas as pd
df = pd.read_csv(paths['data'], parse_dates = ['datetime'])
dt = pd.to_datetime('2010-11-16 18:25')
sst = 12.87
np.where(df.datetime==dt)[0]


blah = (pd.to_datetime(cd.dateloc['datetime'].values).astype(np.int64) / int(1e6)).astype(np.int64)
