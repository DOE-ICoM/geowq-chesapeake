import os
import pandas as pd

from figures import plot_helpers as ph

# Import the reduced data frame and createa geopandas dataframe. Could skip
# if importing the geodataframe directly. Assumes that the dataframe has a modis pixel id column
filename = 'data/unique_pixeldays_w_bandvals.csv'
pix_ids = [274463844, 270230367
           ]  # if you want to plot time series from individual modis pixels
columns = ['SST (C)', 'SSS (psu)', 'turbidity (NTU)']
startDate = '2002-01-01 00:00:00'
endDate = '2021-01-01 00:00:00'
extents = [-77.458841, -74.767094, 36.757802, 39.920274
           ]  #polygon bounding box of our aoi (chesapeake_delaware.shp)

# load dataframe will have to tweak things once we get what the actual dataframe will look like
#gdf = load_dataframe(filepath,filename,datetime_col,sat_pix_col)
filepath = os.environ["icom_data"] + "/Modeling Data/Processed Data p1/"
filename = "aggregated.csv"
df = pd.read_csv(filepath + filename)
df = ph.split_pixelday(df)  # add 'pix_id' and 'date' columns
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.index = df['datetime']

### Temporal Plots ###
# Number of Modis Pixels containing an insitu observation Each day.
[ph.plot_counts(df, column) for column in columns]

# mean time series of an individual or group of modis pixels
[ph.plot_timeseries(df, pix_ids, column) for column in columns]

### Spatial Plots ###
# This uses cartopy for creating maps within python. Alternatively you can make
# your own maps in QGIS and I've included a QGIS document with all the relavent
# background vector data just add the spatially reduced shapefile saved
# above

# Map the number of days containing at least 1 valid observation in each modis pixel
[
    ph.map_counts(df, column, startDate=startDate, endDate=endDate, num=100)
    for column in columns
]

# map the mean clumn value over a given date range.
[
    ph.map_variable(df, column, startDate=startDate, endDate=endDate)
    for column in columns
]

## Save csv to shapefile
#dfsub = df.loc[startDate:endDate].groupby('pix_id').count()
#dfsub.longitude, dfsub.latitude = modisLonLat(dfsub.index)
#gdf = gpd.GeoDataFrame(dfsub, geometry=gpd.points_from_xy(dfsub.longitude, dfsub.latitude),crs = CRS.from_proj4(modis_proj4))
#gdf = gdf.to_crs(epsg = 4326)
#gdf.to_file(savepath + 'unique_pixeldays.shp')
