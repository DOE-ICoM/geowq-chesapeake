# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:40:18 2020

@author: Jon
"""

# Aggregate by pixel-day
import pandas as pd
import numpy as np


path_obs = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data_w_pixel_ids.csv"
# path_obs = r"C:\Users\Jon\Desktop\Research\ICoM\Data\all_data_w_pixel_ids_sampled.csv"
path_pixeldaygrouped = r"C:\Users\Jon\Desktop\Research\ICoM\Data\unique_pixeldays.csv"

# Load dataset and replace -9999 with nans
obs = pd.read_csv(path_obs)
obs.replace(-9999., np.nan, inplace=True)

# Filter by time of day
valid_hrs = [17.1, 19.2]
datetimes = pd.DatetimeIndex(obs.datetime.values)
decimal_hours = datetimes.hour + datetimes.minute/60
# Add a pixel-day identifier for grouping
obs['pixelday'] = [str(pid) + '_' + str(y) + '{:02d}'.format(m) + '{:02d}'.format(d) for pid, y, m, d in zip(obs['pix_id'].values, datetimes.year, datetimes.month, datetimes.day)]
obs = obs[np.logical_and(decimal_hours>valid_hrs[0], decimal_hours<valid_hrs[1])]

# Filter by depth so only "surface" samples are kept
threshold_depth = 1
obs = obs[obs['depth (m)'] <= threshold_depth]


# Do the grouping
def res_agg(x):
    
    def nanmean_wrapper(a):
        """
        Returns np.nan if all the values are nan, else returns the mean.
        """
        if np.isnan(a).all():
            return np.nan
        else:
            return np.nanmean(a)

    
    aggs = {
        'datetime' : lambda x: x['datetime'].values[0],
        'latitude' : lambda x: x['latitude'].values[0],
        'longitude' : lambda x: x['longitude'].values[0],
        'depth (m)' : lambda x: nanmean_wrapper(x['depth (m)']),
        'SSS (psu)' : lambda x: nanmean_wrapper(x['SSS (psu)']),
        'SSS count' : lambda x: np.sum(np.isnan(x['SSS (psu)'])==0),
        'SST (C)' : lambda x: nanmean_wrapper(x['SST (C)']),
        'SST count' : lambda x: np.sum(np.isnan(x['SST (C)'])==0),
        'turbidity (NTU)' : lambda x: nanmean_wrapper(x['turbidity (NTU)']),
        'turbidity count' : lambda x: np.sum(np.isnan(x['turbidity (NTU)'])==0),
        'loc_id' : lambda x: x['loc_id'].values[0],
        }

    do_agg = {k : aggs[k](x) for k in aggs.keys()}

    return pd.Series(do_agg, index=list(aggs.keys())) # This is only guaranteed in Python 3.6+ because dictionaries are ordered therein
avged_by_pixelday = obs.groupby(by='pixelday').apply(res_agg) 
avged_by_pixelday.reset_index(inplace=True) # This "flattens" the indexing

# Add GEE time field for quicker filtering - basically just converting to milliseconds since UNIX epoch
avged_by_pixelday['system:time_start'] = (pd.to_datetime(avged_by_pixelday['datetime'].values).astype(np.int64) / int(1e6)).astype(np.int64)

avged_by_pixelday.to_csv(path_pixeldaygrouped, index=False)

# Debugging/checking data
pd = '262195357_20080918'
blah = obs[obs.pixelday.values==pd]
blah2 = avged_by_pixelday[avged_by_pixelday.pixelday.values==pd]
