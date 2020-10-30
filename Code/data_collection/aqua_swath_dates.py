# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:02:42 2020

@author: Jon

This code takes a spreadsheet created by scraping the https://oceancolor.gsfc.nasa.gov/cgi/browse.pl
page with a search for 'ChesapeakeBay' area. A script was run in the inspection
console to export all the links on the page (note that number of results was
                                             set to 10000 to capture all the
                                             swaths).

See https://oceancolor.gsfc.nasa.gov/browse_help/search_results/ for filename
conventions.

Note that exported times are in UTC (equivalent to GMT).
"""

import pandas as pd
from datetime import datetime, timedelta

path_names = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\MODIS-Aqua\all_chesapeake_swaths_oceancolor.xlsx"
path_out = r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\MODIS-Aqua\aqua_chesapeake_overpass_times.csv"

name_df = pd.read_excel(path_names)
names = name_df.Name.values
names = names[~pd.isna(names)]
names = [n for n in names if '.nc' in n]

# Parse the filenames to a datetime format
dt = []
for name in names:
    year = int(name[1:5])
    doy = int(name[5:8])
    hour = int(name[8:10])
    minute = int(name[10:12])

    dt.append(datetime(year, 1, 1) + timedelta(doy - 1) + timedelta(hours=hour) + timedelta(minutes=minute))

# Determine the unique days
uniques = {(d.year, d.month, d.day) for d in dt}
uniques = sorted(list(uniques))

# Make dataframe where each unique date contains the observation times
starts = []
ends = []
n = []
fnames = []
for u in uniques:
    matches = []
    matchnames = []
    for i, d in enumerate(dt):
        if d.year==u[0] and d.month==u[1] and d.day==u[2]:
            matches.append(d)
            matchnames.append(names[i])
    n.append(len(matches))
    sortem = sorted(matches)
    starts.append(sortem[0].hour + sortem[0].minute/60)
    ends.append(sortem[-1].hour + sortem[-1].minute/60)
    fnames.append(matchnames)

out = pd.DataFrame(data={'date': uniques, 'start': starts, 'end': ends, 'n':n, 'filenames':fnames})
out.to_csv(path_out, index=False)

# Make a histogram of first and last overpass times per day
from matplotlib import pyplot as plt
plt.close('all')
plt.hist(starts)
plt.hist(ends)
plt.legend(['First overpass', 'Last overpass'])
plt.xlabel('Hour of day (UTC)')
plt.ylabel('N overpasses')
