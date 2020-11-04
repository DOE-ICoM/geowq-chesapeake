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

def reduceFeatCol(featCol,indexCol):  
    ''' Spatilaly average the data by stepping through each feature date in the
    date collection, filtering the in-situ data for that date and create a new 
    feat to return populated by the averages for each of the variables. Takes the
    geometry/date from the first feature in the in-situ data feature collection '''
    def mapFeatCol(feat):
        subset = featCol.filter(ee.Filter.eq('pixelDateID',feat.get('pixelDateID')))
        newFeat = ee.Feature(subset.first().geometry(),{})
        newFeat = newFeat.set({'SST (C)' : subset.aggregate_mean('SST (C)'),
                            'SSS (psu)' : subset.aggregate_mean('SSS (psu)'),
                            'turbidity (NTU)': subset.aggregate_mean('turbidity (NTU)'),
                            'count' : subset.aggregate_count('pixelDateID'),
                            'pix_id' : subset.first().get('pix_id'),
                            'dateNumberYMD' : subset.first().get('dateNumberYMD'),
                            'pixelDateID' : feat.get('pixelDateID')})
        return newFeat
    return indexCol.map(mapFeatCol)

def reduceFeature(feat):
    ''' Extract modis pixel information fora given lon/lat point. Note that
    the date here is specific to this script'''
    date = ee.Date(feat.get('dateNumberYMD'));
    image = IC.filterDate(date).first()
    data = image.reduceRegion(ee.Reducer.first(),feat.geometry(),500)
    return ee.Feature(feat.setMulti(data))


#def setGeometry(feat):  
#    ''' adds geometry information to a feature using the longitude and latitude columns. 
#    Assumes the latitude and longitude are points (Only needed when spatially
#    reducing using the reduceColumns function())'''    
#    point = ee.Geometry.Point(ee.List([feat.get('longitude'), feat.get('latitude')]))    
#    return ee.Feature(point).copyProperties(feat)
#
#
#
#def getLonLat(feat):
#    ''' Adds the latitude and longitude as columns to the asset taken from the geometry information.
#    Asssumes the geometry is in the form of a Point(). (Only needed when spatially
#    reducing using the reduceColumns function())'''
#    
#    feat = feat.set('longitude',feat.geometry().coordinates().get(0))
#    feat = feat.set('latitude',feat.geometry().coordinates().get(1))
#    return feat
#
#def join_fc_properties(fc1, fc2, field_match='system:index'):
#    """
#    Combines the properties of two FeatureCollections that contain the same
#    features. The features are defined by "system:index" property.
#    """
#    
#    def combine_properties(feature):
#        newfeat = ee.Feature(feature.get('primary')).copyProperties(feature.get('secondary'))
#        return newfeat
#            
#    fc_filter = ee.Filter.equals(
#            leftField = field_match,
#            rightField = field_match)
#       
#    join = ee.Join.inner('primary', 'secondary')
#    do_join = join.apply(fc1, fc2, fc_filter)
#    
##    clean_join = do_join.map(combine_properties, opt_dropNulls=True)
#    clean_join = do_join.map(combine_properties)
#    
#    return clean_join
#
#
#def getValuesMean(nestedDict,colSelectors,groupName):
#    ''' Convert the nested dictionary outputed by the reduceColumns() function
#    using a mean operator into a feature. Could generalize to other reduction
#    operators though some (such as the count operator) ouput different 
#    dictionary structures'''
#    def getValues(eeDict):        
#        values = ee.Dictionary(eeDict).values([groupName,'mean']).flatten()
#        newDict = ee.Dictionary.fromLists(colSelectors,values)
#        return ee.Feature(None, newDict)
#    return nestedDict.map(getValues)
#
#
#def getValuesCount(eeDict):
#    ''' Convert the dictionary outputed by the reduceColumns() using the count operator into a feature'''
#    return ee.Feature(None, eeDict)

    

    
### Actual Code to Run ####

modis_assets = {'daily_500m' : "MODIS/006/MYD09GA",
                'daily_250m' : 'MODIS/006/MYD09GQ',
                'daily_1000m' : 'MODIS/006/MYDOCGA'}


params = params = {
        'asset' : 'daily_1000m',
        'window': [17.1,19.3],  # valid day times in decimal hours between 0 and 24
        'max_depth': 1,  # Maximum depth of in-situ observation s
        'scale_meters': 1000,  # Modis scale
        'output_dir': 'ICOM',
        'name_field': 'pixelValues_1000m',
        'selectors' : ['SST (C)','SSS (psu)','turbidity (NTU)']
        #'bands': ['NDVI', 'EVI'], Currently just getting all the associated bands

        }


# Didn't upload the assect with the proper date format so gott reformat and add longitude and latitude as columns
stations = ee.FeatureCollection('users/jonschwenk/del_chk_insitu_obs').map(formatDate)
stations = stations.filter(ee.Filter.lessThan('depth (m)',params['max_depth']))

# reduce in-situ observations temporally
hourFilter = ee.Filter.rangeContains('hour', params['window'][0], params['window'][1])
stations = stations.filter(hourFilter).map(pixelDateID)
stations = ee.FeatureCollection(stations.toList(1000, 1000)) # Subset data for testing purposes

# reduce in-situ observations spatially (this is where it times-out)
pixelDateIDs = stations.distinct(['pixelDateID'])
reducedFeatures = reduceFeatCol(stations,pixelDateIDs)

# Original attempt using the reduceColumns() algorithim with the group() method
## Average in-situ observations spatially. NOTE that all columns must be numeric.
#groupName = 'pixelDateID' # column to group observations by when averaging. Can generalize this later
#colSelectors = params['selectors'].copy() # hard-copy pointer reference
#colSelectors.insert(0,groupName)
#colSelectors.extend(['dateNumberYMD','pix_id','longitude','latitude']) # columns to average over
#nCols = len(colSelectors)-1 # number of columns. Needed as ee requires you to repeat the operation each time
#
## Average over observation values in each unique pixel-YMD combination.  
#average = stations.reduceColumns(selectors = colSelectors,reducer = ee.Reducer.mean().repeat(nCols).group(groupField = 0, groupName = groupName))
#average = ee.FeatureCollection(getValuesMean(average.values(['groups']).flatten(),colSelectors,groupName))
## Counts requres an arbitrary numeric column 
#counts = stations.reduceColumns(selectors = [groupName,colSelectors[1]],reducer = ee.Reducer.count().group(groupField = 0, groupName = groupName))    
#counts = ee.FeatureCollection(counts.values(['groups']).flatten().map(getValuesCount))
#
## Combine the feature collections containing the average values and counts 
#reducedFeatures = join_fc_properties(average,counts,groupName).map(setGeometry)
#

# Extract pixel information
IC = ee.ImageCollection(modis_assets[params['asset']])
output = reducedFeatures.map(reduceFeature)
#selectors = output.first().propertyNames()  # Columns to keep when exporting. Taking them all atm

# Download the dataframe
task = ee.batch.Export.table.toDrive(
  collection = output,
  description = params['name_field'],
  fileFormat = 'CSV',
  folder = params['output_dir'],
)

task.start()
