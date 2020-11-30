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
hours_filt = [14.1, 22.3] # which hours of the day to filter the observations to
# hours_filt = [14.1, 20] # which hours of the day to filter the observations to
depth_filt = [0, 1] # depth range to keep
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
# will fail for large datasets (i.e. > 20,000 rows)
cd.start_gee_bandval_retrieval(gdrive_folder=gdrive_folder)
# Failed, so we upload the shapefile manually and pass in the asset location
gee_asset = 'users/jonschwenk/aggregated_gee_3hrwindow'
cd.start_gee_bandval_retrieval(asset=gee_asset, gdrive_folder=gdrive_folder)

"""Need to wait for GEE task to finish and download the .csv"""
path_bandvals = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Processed Data\unique_pixeldays_w_bandvals.csv"
cd.merge_bandvals_and_data(path_bandvals)
cd.aggregated.to_csv(r"C:\Users\Jon\Desktop\Research\ICoM\Data\Processed Data\aggregated_w_bandvals.csv", index=False)

# Compute data availability for specific variables
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

QCcols = [k for k in cd.aggregated.keys() if 'QC' in k]
var = 'SSS (psu)'
var = 'SST (C)'
var = 'turbidity (NTU)'

valid_data = [0] # which QC values are acceptable data
dftemp = cd.aggregated[~pd.isna(cd.aggregated[var])]
q_array = dftemp[QCcols].to_numpy()
q_valid = np.zeros(q_array.shape)
for v in valid_data:
    q_valid[q_array==v] = 1
rowct_valid = q_valid.sum(axis=1)
plt.close()
plt.hist(rowct_valid, bins=np.arange(0,11)-0.5, width=0.9)
plt.xlabel('# good bands for an observation')
plt.ylabel('count')
plt.title(var)



