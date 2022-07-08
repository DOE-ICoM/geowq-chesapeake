# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:36:11 2020

@author: Jon
"""

import ee
ee.Initialize()

def getBandValues(feature):
    
    # Format date for filtering (this should be done in the asset)    
    datetime = ee.Date.parse("yyy-MM-dd HH:mm:ss", ee.String(feature.get('datetime')))
    dateYMD = datetime.format('yyyy-MM-dd')
    
    # Filter the imageCollection to the date of the feature
    icFilt = ic.filterDate(ee.Date(dateYMD), ee.Date(dateYMD).advance(1, 'day'))
    
    # Initialize the feature to return
    newFeat = ee.Feature(feature.geometry(), {'pixelday' : feature.get('pixelday')})

    # For some reason, the date filtering sometimes returns an empty imageCollection
    # Use an if statement to return a feature with all null properties in those cases
    newFeat = ee.Algorithms.If(
        icFilt.size(), # If this is 0, we have no images in the collection and it evaluates to "False", else "True"
        newFeat.setMulti(icFilt.first().reduceRegion(ee.Reducer.first(), feature.geometry(), 500)), #  True: grab its band values
        newFeat # False: only geometry and 'pixelday' properties will be set
    )

    return newFeat

def combineJoin(joinedFeat):
    f1 = ee.Feature(joinedFeat.get('primary'))
    f2 = ee.Feature(joinedFeat.get('secondary'))
    return f1.set(f2.toDictionary())

# Define some modis asset locations
modis_assets = {'daily_500m' : "MODIS/006/MYD09GA",
                'daily_250m' : 'MODIS/006/MYD09GQ',
                'daily_1000m' : 'MODIS/006/MYDOCGA'}

assets = {'uniquePixelDays' : 'users/jonschwenk/chk_del_unique_pixeldays'}

# Set parameters of analysis
params = params = {
        'asset' : 'daily_1000m',
        'scale_meters': 1000,  # scale to perform analysis (1000 for MYD09GCA)
        'output_dir': 'ICOM',
        'name_field': 'del_chk_obs_w_modis_bands',
        # 'selectors' : ['SST (C)','SSS (psu)','turbidity (NTU)']
        }

# Need to set the 'id' field as pixelday
# Have a system:time_start column (nanoseconds since epoch) AND a yyyy-mm-dd column
# Not sure yet about nans/Nones

# Load the imageCollection, get date range
ic = ee.ImageCollection(modis_assets[params['asset']])
icDateRange = ic.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])

# Load the data, filter it to the imageCollection dateRange
obs = ee.FeatureCollection(assets['uniquePixelDays'])
# # TEMPORARY: change the column name of pixelday to id -- this should be fixed in the data export (before running this script)
# def renameObs(feature):
#      rnFeat = ee.Feature(feature.geometry(), {'id':feature.get('pixelday')})
#      rnFeat = rnFeat.copyProperties(feature, exclude=['pixelday'])
#      return rnFeat
# obs = obs.map(renameObs)
# blah = obs.first().getInfo()


obs = obs.filterDate(ee.Date(icDateRange.get('min')), ee.Date(icDateRange.get('max')))

# obs = ee.FeatureCollection(obs.toList(100, 1000))
fcBandVals = obs.map(getBandValues)
# blah = fcBandVals.first().getInfo()

# Append the bandVals to the obs featureCollection
filt = ee.Filter.equals(
  leftField = 'pixelday',
  rightField = 'pixelday')
join = ee.Join.inner();
joined = join.apply(obs, fcBandVals, filt)
# Combine the columns of the joined FC
joined = joined.map(combineJoin)
     
# Export the dataframe
task = ee.batch.Export.table.toDrive(
  collection = joined,
  description = params['name_field'],
  fileFormat = 'CSV',
  # folder = params['output_dir'],
)

task.start()


