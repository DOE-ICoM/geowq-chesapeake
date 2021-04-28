# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:50:19 2020

@author: Jon
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

path_bv = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Processed Subset\aggregated.csv"

df = pd.read_csv(path_bv)

pixel_ids = np.array([int(pd.split('_')[0]) for pd in df.pixelday.values])

pid = 270230367

dfp = df.iloc[np.where(pixel_ids==pid)[0]]
dfp = dfp.sort_values(by='datetime')
dates = pd.to_datetime(dfp['datetime']).values

plt.close('all')
fig, ax = plt.subplots()
ax.plot(dates, dfp['SST (C)'])
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
fig.autofmt_xdate()

path_filt = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Processed Subset\filtered.csv"
dff = pd.read_csv(path_filt)
dffp = dff[dff.loc_id==46922]

dffp = dffp.sort_values(by='datetime')
dates = pd.to_datetime(dffp['datetime']).values

plt.close('all')
fig, ax = plt.subplots()
ax.plot(dates, dfp['SST (C)'])
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
fig.autofmt_xdate()


path_singlebuoy = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Buoys\buoy_dataframes\44042.csv"
dfb = pd.read_csv(path_singlebuoy)
dfb = dfb.sort_values(by='Unnamed: 0')
dates = pd.to_datetime(dfb['Unnamed: 0']).values

plt.close('all')
fig, ax = plt.subplots()
ax.plot(dates, dfb['SST (C)'])
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
fig.autofmt_xdate()
