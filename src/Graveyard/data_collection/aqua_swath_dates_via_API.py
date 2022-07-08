# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:43:12 2020

@author: Jon

Screwing around with API to retrieve overpass times. There are limits to the
size of the URL request.

This script is unfinished but not sure if we need it at this point.
"""

import geopandas as gpd
import requests

# Get bounding box coordinates for target area
gdf = gpd.read_file(r"C:\Users\Jon\Desktop\Research\ICoM\satval\Data\Boundaries\delaware_chesapeake.shp")
bb = gdf.geometry.values[0].bounds

# Request MODIS file ids
modis_product = 'MYDOCGA'
start_date = '2018-01-01' # YYYY-MM-DD
end_date = '2019-01-01' # YYYY-MM-DD
url_get_fileIDs = r'https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/searchForFiles?product={}&start={}&stop={}&north={}&south={}&west={}&east={}&coordsOrTiles=coords'.format(modis_product, start_date, end_date, bb[3], bb[1], bb[0], bb[2])
response = requests.get(url_get_fileIDs)

# Extract MODIS file ids from return string
rlist = response.text.split('<return>')
rlist.pop(0)
fileIDs = []
for r in rlist:
    fileIDs.append(r.split('</return>')[0])

# Make second request to get file names which contain time information
filestring = ''
for i, f in enumerate(fileIDs):
    filestring = filestring + f + ','
    if i > 500:
        break
filestring = filestring[:-1]
url_filenames = 'https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/getFileProperties?fileIds={}'.format(filestring)
response = requests.get(url_filenames)
print(response)

raw_fns = response.text.split('<mws:fileName>')
raw_fns.pop(0)
filenames = [f.split('</mws:fileName>')[0] for f in raw_fns]