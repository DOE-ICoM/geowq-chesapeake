# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:28:09 2020

@author: Jon
"""

import ee
ee.Initialize()


def getBandValues(im):
    
    # Filter the observations to the date range of the image
    obsSameDate = obs.filterDate(ee.Date(im.get('system:time_start')), ee.Date(im.get('system:time_end')))

    bandValFC = ee.Algorithms.If(
        obsSameDate.size(), # If this is 0, we have no images in the collection and it evaluates to "False", else "True"
        im.reduceRegions(obsSameDate, ee.Reducer.first(), scale=500), #  True: grab its band values
        None # False: only geometry and 'pixelday' properties will be set
    )

    return bandValFC

# Define some modis asset locations
modis_assets = {'daily_500m' : "MODIS/006/MYD09GA",
                'daily_250m' : 'MODIS/006/MYD09GQ',
                'daily_1000m' : 'MODIS/006/MYDOCGA'}

assets = {'uniquePixelDays' : 'users/jonschwenk/satval_test_novars'}

# Set parameters of analysis
params = {
        'asset' : 'daily_1000m',
        'scale_meters': 1000,  # scale to perform analysis (1000 for MYD09GCA)
        'output_dir': 'ICOM',
        'name_field': 'del_chk_obs_w_modis_bands_testall',
        # 'selectors' : ['SST (C)','SSS (psu)','turbidity (NTU)']
        }

# Load the imageCollection, get date range
ic = ee.ImageCollection(modis_assets[params['asset']])
icDateRange = ic.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])

# Load the observation datelocs, filter them to the imageCollection dateRange
obs = ee.FeatureCollection(assets['uniquePixelDays'])
obs = obs.filterDate(ee.Date(icDateRange.get('min')), ee.Date(icDateRange.get('max')))

# ic = ee.ImageCollection(ic.toList(10, 1000))

bandVals = ic.map(getBandValues, opt_dropNulls=True).flatten()

     
# Export the dataframe
task = ee.batch.Export.table.toDrive(
  collection = bandVals,
  description = params['name_field'],
  fileFormat = 'CSV',
  # folder = params['output_dir'],
)

task.start()

