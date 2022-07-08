# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:31:30 2020

@author: muklu
"""
import ee
ee.Initialize()

def formatDate(feat):
     datetime = ee.Date.parse("yyy-MM-dd HH:mm:ss", ee.String(feat.get('datetime')))
     
     # Compute dcimal hour for filtering out in-situ data
     hour = ee.Number.parse(datetime.format('HH'))
     minutes = ee.Number.parse(datetime.format('mm'))
     seconds = ee.Number.parse(datetime.format('ss'))
     decimalHour = hour.add(minutes.divide(60)).add(seconds.divide(60*60))
     
     # Get the Yyear-mont-day for each datetime, used to reduce the in-situ data
     dateYMD = datetime.format('yyyy-MM-dd')
     dateNumber = ee.Date.parse('yyy-MM-dd',dateYMD).millis()
     systemDate = datetime.millis() #incase we use other products in the future
     
     # Save to feature
     return feat.set({'datetime': datetime,'hour': decimalHour,'date':dateYMD,'system:time_start':systemDate, 'dateNumberYMD':dateNumber})

def pixelDateID(feat):
    ''' Create a unique string id based on the modis pixel ID and the YMD date.
    this is used for the spatial averaging of points. '''
    
    dateString = ee.Date(feat.get('datetime')).format('yyyy-MM-dd')
    pixel_id = ee.Algorithms.String(feat.get('pix_id')).cat('_')
    return feat.set('pixelDateID',pixel_id.cat(dateString))

def uniquifyAndAddBandvals(ft):

    # The feature contains a property called "matches" that contains a list
    # of observations that map to the unique pixelday; gotta package it as an FC
    matchesFC = ee.FeatureCollection(ee.List(ft.get('matches')))

    # Get the averages and counts
    properties = ['SST (C)', 'SSS (psu)', 'turbidity (NTU)']
    means = matchesFC.filter(ee.Filter.notNull(properties)).reduceColumns(
      reducer = ee.Reducer.mean().repeat(3),
      selectors = properties)
    counts = matchesFC.filter(ee.Filter.notNull(properties)).reduceColumns(
      reducer = ee.Reducer.count().repeat(3),
      selectors = properties)
    
    # Get the image to pull band values from
    image = IC.filterDate(ee.Date(ft.get('dateNumberYMD'))).first()
    data = image.reduceRegion(ee.Reducer.first(),ft.geometry(),500)
    
    # Store aggregations in new feature
    newFeat = ee.Feature(ft.geometry(),{})   
    newFeat = newFeat.set({'SST (C)' : ee.List(means.get('mean')).get(0),
                           'SSS (psu)' : ee.List(means.get('mean')).get(1),
                           'turbidity (NTU)' : ee.List(means.get('mean')).get(2),
                           'SST (C) count' : ee.List(counts.get('count')).get(0),
                           'SSS (psu) count' : ee.List(counts.get('count')).get(1),
                           'turbidity (NTU) count' : ee.List(counts.get('count')).get(2),
                           'pixelDateID' : ft.get('pixelDateID'),
                           'dateNumberYMD' : ft.get('dateNumberYMD')
                           })
    # Store the band values
    newFeat = newFeat.setMulti(data)

    return newFeat


# Define some modis asset locations
modis_assets = {'daily_500m' : "MODIS/006/MYD09GA",
                'daily_250m' : 'MODIS/006/MYD09GQ',
                'daily_1000m' : 'MODIS/006/MYDOCGA'}

# Set parameters of analysis
params = {
        'asset' : 'daily_1000m',
        'window': [17.1,19.3],  # valid day times in decimal hours between 0 and 24
        'max_depth': 1.01,  # Maximum depth of in-situ observation s
        'scale_meters': 1000,  # Modis scale
        'output_dir': 'ICOM',
        'name_field': 'del_chk_obs_w_modis_bands',
        'selectors' : ['SST (C)','SSS (psu)','turbidity (NTU)']
        #'bands': ['b1', 'b2'], Currently just getting all the associated bands
        }


# Load observations and format their dates
observations = ee.FeatureCollection('users/jonschwenk/del_chk_insitu_obs')

# We only include observations near the surface of the ocean
observations = observations.filter(ee.Filter.lessThan('depth (m)',params['max_depth']))

# Format the dates
observations = observations.map(formatDate)

# MODIS overpass times range from  ~17:00 to ~19:00 for our study region, so
# filter out the observations outside this window 
hourFilter = ee.Filter.rangeContains('hour', params['window'][0], params['window'][1])
observations = observations.filter(hourFilter).map(pixelDateID)

# Subset the observations to reduce computational burden and/or for testing
# observations = ee.FeatureCollection(observations.toList(1000, 10000))
# observations.first().getInfo()

# Use a join to generate the unique pixelDateIDs FC that contains an index.
# This prevents the re-searching of the full FC every time we filter to look
# for a given pixeldate.
uniquePixelDateIDs = observations.distinct(['pixelDateID'])
pixelDateIDs = ee.Join.saveAll(matchesKey="matches", outer=True).apply(uniquePixelDateIDs, observations, ee.Filter.equals('pixelDateID', None, 'pixelDateID'))

# Retrieve band values for each pixel/day
IC = ee.ImageCollection(modis_assets[params['asset']])
   
avged = pixelDateIDs.map(uniquifyAndAddBandvals)

# Download the dataframe
task = ee.batch.Export.table.toDrive(
  collection = avged,
  description = params['name_field'],
  fileFormat = 'CSV',
  # folder = params['output_dir'],
)

task.start()


# Formatting dates for GEE ingestion
import pandas as pd

df = pd.read_csv(r"C:\Users\Jon\Desktop\Research\ICoM\Data\unique_pixeldays_test.csv")
dts = [pd.to_datetime(dt) for dt in df.datetime.values]
ymd = [str(d.year)+'-'+str(d.month)+'-'+str(d.day) for d in dts]
df['datetime'] = ymd
df = df.iloc[:100]
df.to_csv(r"C:\Users\Jon\Desktop\Research\ICoM\Data\unique_pixeldays_ymd.csv", index=False)

