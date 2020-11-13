# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:24:41 2020

@author: muklu
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import CRS


modis_proj4 = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs' # pulled from GEE
def modisLonLat(p_idx):
    # Some MODIS grid info for MYDOCGA
    modis_proj4 = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs' # pulled from GEE
    modis_gt = (926.625433056, 0, -20015109.354, 0, -926.625433055, 10007554.677) # geotransform pulled from GEE: (xScale, xShearing, xTranslation, yShearing, yScale, yTranslation)
    modis_shape = (43200,21600) # ncols, nrows

    r, c = np.unravel_index(p_idx, modis_shape[::-1])
    lons = modis_gt[2] + modis_gt[0] * c + modis_gt[0]/2
    lats = modis_gt[5] + modis_gt[4] * r + modis_gt[4]/2
    return lons,lats




# Import the reduced data frame and createa geopandas dataframe. Could skip
# if importing the geodataframe directly. Assumes that the dataframe has a modis pixel id column
filepath = 'F:/ICOM/'
df = pd.read_csv(filepath + 'all_data_reduced.csv')
df['datetime'] = pd.to_datetime(df['datetime'],format = '%Y-%m-%d %H:%M:%S')
df.index = df['datetime']
df['date'] = pd.to_datetime(df['datetime'].dt.strftime('%Y-%m-%d'))
df.longitude,df.latitude = modisLonLat(df['pix_id'].values)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude),crs = CRS.from_proj4(modis_proj4))
gdf = gdf.to_crs(epsg = 4326)
gdf['longitude'] = gdf.geometry.x
gdf['latitude'] = gdf.geometry.y 



# things to adjust
savepath = 'F:/ICOM/'
columns = ['SST (C)','SSS (psu)','turbidity (NTU)']
startDate = pd.to_datetime('2001-01-01')
endDate = pd.to_datetime('2021-01-01')
extents = [-77.458841,  -74.767094, 36.757802, 39.920274] #polygon bounding box of our aoi (chesapeake_delaware.shp)



# Number of Modis Pixels containing an insitu observation Each day
for column in columns:
    gdfsub = gdf[~pd.isna(df[column])].groupby('date')['pix_id'].nunique().loc[startDate:endDate]   
    # plot data
    plt.figure(figsize = (15,10))
    plt.title('# of Pixels Containing Observations: ' + column,fontsize = 40)
    plt.plot(gdfsub.index,gdfsub.values,'.k',markersize =6)
    plt.xlabel('Date',fontsize=32)
    plt.ylabel('Count',fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)





# Number of Observations per modis pixel (Note that the count method ignores nan values)
gdfsub = gdf.groupby(['pix_id','date'])[columns].mean().groupby(level=0).count()
gdfsub.longitude, gdfsub.latitude = modisLonLat(gdfsub.index.values)
gdfsub = gpd.GeoDataFrame(gdfsub, geometry=gpd.points_from_xy(gdfsub.longitude, gdfsub.latitude),crs = CRS.from_proj4(modis_proj4))
gdfsub = gdfsub.to_crs(epsg = 4326)
gdfsub.longitude = gdfsub.geometry.x
gdfsub.latitude = gdfsub.geometry.y 
gdfsub.to_file(savepath + 'counts_spatial') #save as shapefile

# If plotting data as well
# This uses cartopy for creating maps within python. Alternatively you can make 
# your own maps in QGIS and I've included a QGIS document with all the relavent 
# background vector data just add the spatially reduced shapefile saved
# above 

import cartopy.crs as ccrs
import cartopy
from cartopy.feature import NaturalEarthFeature, COLORS


ocean = NaturalEarthFeature(category='physical', name='ocean',
                            scale='10m', facecolor=COLORS['water'])

land = NaturalEarthFeature(category='physical', name='land',
                            scale='10m', facecolor=COLORS['land'])

states_provinces = NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none',
        edgecolor= 'gray')

# Plot each variable in the reduced dataframe
for column in columns:
    plt.figure(figsize=[15,15])
    
    # create dummy plot in order to get information for creating the legend
    ax = plt.scatter(gdfsub.longitude,gdfsub.latitude,s = gdfsub[column],zorder=2)
    handles, labels = ax.legend_elements(prop="sizes", num=100,  markerfacecolor='none', markeredgecolor='k')
    
    # set up the actual background map
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extents)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(land)
    ax.add_feature(ocean)
    ax.add_feature(states_provinces,linestyle=':')
    # manuallly add come state labels
    ax.text(-76.98, 39.56, 'Maryland', transform=ccrs.Geodetic(),color ='gray')
    ax.text(-76.98, 39.81, 'New York', transform=ccrs.Geodetic(),color ='gray')
    ax.text(-75.1, 39.56, 'New Jersey', transform=ccrs.Geodetic(),color ='gray')
    ax.text(-75.5, 38.7, 'Delaware', transform=ccrs.Geodetic(),color ='gray')
    ax.text(-77.16, 36.93, 'Virginia', transform=ccrs.Geodetic(),color ='gray')
    
    # plot the count information and add a rough legend
    ax.scatter(gdfsub.longitude,gdfsub.latitude,s = gdfsub[column],zorder=2, facecolors='none', edgecolors='k')
    ax.legend(handles[::25], labels[::25], loc="lower right", title="Number of \n Observations",handletextpad=3.5,labelspacing=1.5,borderpad=3)
    plt.title('Total Number of Observations: ' + column)
