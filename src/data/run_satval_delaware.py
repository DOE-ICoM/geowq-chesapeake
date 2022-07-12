# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:53:35 2020

@author: Jon

env: icomdata
"""
import sys
sys.path.append(r'C:\Users\Jon\Desktop\Research\ICoM\satval\Code')
from satval_class import satval
import pandas as pd
import numpy as np

path_params = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Delaware P1\params.csv"
cd = satval(path_params)

""" Processing begins here. """
cd.assign_unique_location_ids()
cd.map_coordinates_to_pixels()
cd.aggregate_data_to_unique_pixeldays()

# Try to start a GEE task by directly uploading the geodataframe - this 
# will fail for large datasets (e.g. > 20,000 rows)
cd.start_gee_bandval_retrieval()
# # Failed, so we upload the shapefile manually and pass in the asset location
# gee_asset = 'users/jonschwenk/aggregated_gee_p1'
# cd.start_gee_bandval_retrieval(asset=gee_asset)

"""Need to wait for GEE task to finish and download the .csv"""
path_bandvals = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Delaware P1\unique_pixeldays_w_bandvals.csv"
cd.merge_bandvals_and_data(path_bandvals)

# Compute data availability for specific variables and considering QC bands
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
all_valid_ct = np.sum(rowct_valid==9)
plt.close()
plt.hist(rowct_valid, bins=np.arange(0,11)-0.5, width=0.9)
plt.xlabel('# good bands for an observation')
plt.ylabel('count')
plt.title(var)


