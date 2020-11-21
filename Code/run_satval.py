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
paths = {}
paths['data'] = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data.csv"
paths['bounding_pgon'] = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\Boundaries\chk_water_only.geojson"
paths['pixel_centers'] = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\MYDOCGA\pixel_centers.shp"
paths['aggregated'] = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data_aggregated.csv"
gee_filename = 'chk_del_full_bandvals'
gdrive_folder = 'ICOM exports'

hours_filt = [17.1, 19.3] # which hours of the day to filter the observations to
depth_filt = [0, 1] # what is the deepest measurement to keep
data_columns = ['SST (C)', 'turbidity (NTU)', 'SSS (psu)', 'depth (m)'] # which data should be kept and aggregated
satellite_data = 'MYDOCGA.006' # GEE ID of dataset we want pixel values from
frac_pixel_thresh = .9 # what fraction of each pixel must be inside the AOI polygon to include it

filters = {'depth (m)': depth_filt, 'time_of_day' : hours_filt}


""" Processing begins here. """
cd = satval(paths, filters=filters)
cd.assign_unique_location_ids()
cd.map_coordinates_to_pixels(dataset=satellite_data, frac_pixel_thresh=frac_pixel_thresh)
cd.aggregate_data_to_unique_pixeldays(datacols=data_columns)
# Export the aggregated dataframe
cd.aggregated.to_csv(paths['aggregated'], index=False)
cd.start_gee_bandval_retrieval(satellite_data, gee_filename, gdrive_folder=gdrive_folder)

