# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:19:04 2020

@author: Jon

Fetching buoy data.

It is important to know the time zones of the reported time. From 
https://www.ndbc.noaa.gov/measdes.shtml, we have:
    
"Time: Station pages show current observations in station local time by default, 
but can be changed by the viewer to UTC (formerly GMT). Both Realtime and Historical 
files show times in UTC only."

This means that downloaded data files are in UTC already; but the individual
buoy page online is converted to local time. The TIMEZONE column is therefore
apparantly only used for the web display.


env: icomdata
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
import os
import numpy as np



""" Get list of all stations within target area """
def llstrparse(string):
    """
    For parsing the lat/lon string in NBDC's table.
    """
    def dmsify(st):
        d = float(st.split('&#176')[0])
        m = float(st.split('&#176')[1].split('\'')[0].strip(';'))
        s = float(st.split('\'')[1].strip('"'))
        dd = d + m/60 + s/3600
        return dd

    s = string.split('(')[1]
    
    if 'S' in s:
        latmult = -1
    else:
        latmult = 1
    
    if 'W' in s:
        lonmult = -1
    else:
        lonmult = 1
        
    lat = s.split(' ')[0]
    lon = s.split(' ')[2]
       
    lat = dmsify(lat) * latmult
    lon = dmsify(lon) * lonmult
    
    return lon, lat    

# Downloaded from https://www.ndbc.noaa.gov/data/stations/
path_allstas = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Buoys\station_table.txt"
stations = pd.read_csv(path_allstas, sep="|")
stations = stations[1:]

# Read in boundaries
path_bd = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Boundaries\delaware_chesapeake.shp"
gdf = gpd.read_file(path_bd)
bgeom = gdf.geometry.values[0]

within = []
geoms = []
lats = []
lons = []
timezone = []
for l, name, tz in zip(stations[' LOCATION '].values, stations['# STATION_ID '], stations[' TIMEZONE ']):   
    
    lon, lat = llstrparse(l)    
    pt = Point(lon, lat)
    geoms.append(pt)
    
    if pt.intersects(bgeom) is True:
        within.append(name)
        lats.append(lat)
        lons.append(lon)
        timezone.append(tz)
    
# blah = gpd.GeoDataFrame(geometry=geoms, crs=CRS.from_epsg(4326))
# blah.to_file(r'C:\Users\Jon\Desktop\Research\ICoM\Data\Buoys\all_buoys.json', driver='GeoJSON')
    
"""
Now that valid stations are known, extract the measured variables at these
stations. 

I am using the NDBC package: https://pypi.org/project/NDBC/
It's an API for fetching data from the ndbc site via http. I had to make
some modifications so it would fetch "ocean" data as well (for salinity), as
well as updates for the different kinds of data that appear in NDBC txt files.
Unfortuntely they're not completely standardized.

Each station thus needs to be run twice: using dtype='stdmet' for SST, and
using dtype='ocean' for salinity. 

Buoy salinity measurements are currently hosted at: 
    [historical] https://www.ndbc.noaa.gov/data/historical/ocean/
    [current year] https://www.ndbc.noaa.gov/data/ocean/
    
Buoy SST measurements are currently hosted at:
    [historical] https://www.ndbc.noaa.gov/data/historical/stdmet/
    [current year] https://www.ndbc.noaa.gov/data/stdmet/
"""

from NDBC.NDBC import DataBuoy
yrmin = 1985
yrmax = 2021
current_yr = 2020
months = range(1,13)

path_baseout = r'C:\Users\Jon\Desktop\Research\ICoM\Data\Buoys\buoy_dataframes'

# Prepare dataframe for aggregating results
# master = pd.DataFrame(index=columns=['datetime', 'station_id', 'longitude', 'latitude', 'SST (C)', 'SSS (psu)'])

for sta, la, lo in zip(within, lats, lons):
    if os.path.isfile(os.path.join(path_baseout, str(sta) + '.csv')) is True:
        continue
    print(sta)
    sss = pd.DataFrame()
    sst = pd.DataFrame()
    
    # Get SST data
    DB = DataBuoy(sta, dtype='stdmet')
    DB.get_stdmet(years=range(yrmin, yrmax), months=months, datetime_index=True)
    if 'data' in DB.data['stdmet'].keys():
        sst = DB.data['stdmet']['data'] # pandas dataframe
        
        # Format SST data
        keepcols = ['WTMP']
        dropcols = [k for k in sst.keys() if k not in keepcols]
        sst = sst.drop(columns=dropcols)
        sst = sst.rename(columns={'WTMP':'SST (C)'})
        sst = sst[~pd.isna(sst['SST (C)'])]
    
    # Get SSS data
    DB = DataBuoy(sta, dtype='ocean')
    DB.get_stdmet(years=range(yrmin, yrmax), months=months, datetime_index=True)
    if 'data' in DB.data['stdmet'].keys():
        sss = DB.data['stdmet']['data']
        
        # Format SSS data
        sss = sss.rename(columns={'DEPTH':'depth (m)', 'SAL':'salinity (PSU)', 'TURB': 'turbidity (FTU)'})
        keepcols = ['depth (m)', 'salinity (PSU)', 'turbidity (FTU)']
        dropcols = [k for k in sss.keys() if k not in keepcols]
        sss = sss.drop(columns=dropcols)
        
        # Merge SST and SSS
        joined = []
        joined = sst.join(sss)
    else:
        joined = sst
    
    # Append station metadata
    joined['latitude'] = np.ones(len(joined)) * la
    joined['longitude'] = np.ones(len(joined)) * lo
    joined['station_id'] = [sta for i in range(len(joined))]
    joined['source'] = ['NDBC' for i in range(len(joined))]
    
    joined.to_csv(os.path.join(path_baseout, str(sta) + '.csv'), index=True)
    
    

