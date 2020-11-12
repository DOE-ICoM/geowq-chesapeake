# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:53:35 2020

@author: Jon
"""
import sys
sys.path.append(r'C:\Users\Jon\Desktop\Research\ICoM\satval\Code')
from satval_class import satval

paths = {}
paths['data'] = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data_subset.csv"
paths['bounding_pgon'] = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\Boundaries\delaware_chesapeake.shp"
paths['pixel_centers'] = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\MYDOCGA\pixel_centers.shp"
paths['aggregated'] = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data_subset_aggregated.csv"

use_hours = [17.1, 19.3] # which hours of the day to filter the observations to
max_depth = 1. # what is the deepest measurement to keep
data_columns = ['SST (C)', 'tubidity (NTU)', 'SSS (psu)']
satellite_data = 'MYDOCGA.006'

cd = satval(paths, filters={'max_depth': 1, 'hours': use_hours})

cd.filter_by(hours=use_hours, depth=max_depth) 

cd.assign_unique_location_ids()
cd.map_locations_to_pixels(dataset = satellite_data)


import modin.pandas as pd
import time
t = time.time()
path_df = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data.csv"
pd.read_csv(path_df, sep=',', usecols=['datetime', 'longitude', 'latitude'], 
                            parse_dates = ['datetime'], squeeze=True)
print(time.time()-t)

import pandas as pd
import time
t = time.time()
pd.read_csv(path_df, sep=',', usecols=['datetime', 'longitude', 'latitude'], 
                            parse_dates = ['datetime'], squeeze=True)
print(time.time()-t)