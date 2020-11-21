# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:53:34 2020

@author: Jon

Download MODIS band values using the appeears API. https://lpdaacsvc.cr.usgs.gov/appeears/api/?language=Python%203#introduction

Following the python API example given here: https://lpdaac.usgs.gov/resources/e-learning/getting-started-with-the-a%CF%81%CF%81eears-api-submitting-and-downloading-a-point-request/

Not sure about request size limits...would be nice to get all the data
in one request.

Ok, there is a 1000 point limit on API requests, and some kind of computed
limit that relates to the time period requested. This means that to fetch the
entire dataset, we'd need to break the unique pixels up into 1000 point chunks,
and then run individual years for each chunk. My guess is that although all
this can be automated, it would be a real headache to ensure that each API
request actually completes properly. Going to try the GEE route instead.
"""
import requests as r
import getpass, pprint, time, os, cgi
import pandas as pd

# Path to the mapped-to-pixel observations
path_obs = r"C:\Users\Jon\Desktop\Research\ICoM\Data\unique_pixeldays.csv"

# Where to store downloads
path_dl = r'C:\Users\Jon\Desktop\Research\ICoM\Data\appeears_tmp'
os.chdir(path_dl)                                      # Change to working directory
api = 'https://lpdaacsvc.cr.usgs.gov/appeears/api/'  # Set the AρρEEARS API to a variable

# Authenticate session
# user = getpass.getpass(prompt = 'Enter NASA Earthdata Login Username: ')
# password = getpass.getpass(prompt = 'Enter NASA Earthdata Login Password: ')  
user = 'schwenk'
password = 'ic0mic0m'
token_response = r.post('{}login'.format(api), auth=(user, password)).json() # Insert API URL, call login service, provide credentials & return json
del user, password                                                           # Remove user and password information

# product_response = r.get('{}product'.format(api)).json()                         # request all products in the product service
# print('AρρEEARS currently supports {} products.'.format(len(product_response)))  # Print no. products available in AρρEEARS
# products = {p['ProductAndVersion']: p for p in product_response} # Create a dictionary indexed by product name & version
# products[prod]                                         # Print information for MCD15A3H.006 LAI/FPAR Product

# Set MODIS product we want to draw from
prod = 'MYDOCGA.006'

# Format the layers/products appropriately. Requesting all bands, including
# quality control ones.
lst_response = r.get('{}product/{}'.format(api, prod)).json()  # Request layers for the 2nd product (index 1) in the list: MOD11A2.006
bands = list(lst_response.keys())                                          # Print the LAI layer names 
all_layers = []
for b in bands:
    all_layers.append({'layer':b, 'product':prod})

# Make request
token = token_response['token']                      # Save login token to a variable
head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit request

task_name = 'del_chk_uniques' # User-defined name of the task 'NPS Vegetation' used here
task_type = ['point']  # Type of task, area or  point
# startDate = '01-01-2001'      # Start of the date range for  which to extract data: MM-DD-YYYY
# endDate = '12-31-2001'        # End of the date range for  which to extract data: MM-DD-YYYY
startDate = '01-01-2001'      # Start of the date range for  which to extract data: MM-DD-YYYY
endDate = '01-01-2021'        # End of the date range for  which to extract data: MM-DD-YYYY
recurring = False             # Specify True for a recurring date range

# Need to build the coordinate dictionary - get a unique list of pixelids and 
# their associated coordinates
obs = pd.read_csv(path_obs)
obs['pixel_ids'] = [int(pid.split('_')[0]) for pid in obs.pixelday.values]
obs_unique_pixels = obs.drop_duplicates(subset=['pixel_ids'])

# Make a list of coordinate dictionaries to request
coordinates = []
for i, row in obs_unique_pixels.iterrows():
    coordinates.append({
        'id' : str(row['pixel_ids']),
        'longitude' : str(row['longitude']),
        'latitude' : str(row['latitude'])        
        })
    if len(coordinates) == 1000:
        break

# Create the task       
task = {
    'task_type': task_type[0],
    'task_name': task_name,
    'params': {
         'dates': [
         {
             'startDate': startDate,
             'endDate': endDate
         }],
         'layers': all_layers,
         'coordinates': coordinates
    }
}

# Submit the task
task_response = r.post('{}task'.format(api), json=task, headers=head).json()  # Post json to API task service, return response as json

params = {'limit': 5, 'pretty': True} # Limit API response to 2 most recent entries, return as pretty json
 
tasks_response = r.get('{}task'.format(api),params = params, headers=head).json() # Query task service setting params & header
tasks_response                                                                    # Print tasks response

task_id = task_response['task_id']                                               # Set task id from request submission
status_response = r.get('{}status/{}'.format(api, task_id), headers=head).json() # Call status service w/ specific task ID & username
status_response                                                                  # Print response


destDir = os.path.join(path_dl, task_name)                # Set up output directory using input directory and task name
if not os.path.exists(destDir):os.makedirs(destDir)     # Create the output directory

bundle = r.get('{}bundle/{}'.format(api,task_id)).json()  # Call API and return bundle contents for the task_id as json

files = {}                                                       # Create empty dictionary
for f in bundle['files']: files[f['file_id']] = f['file_name']   # Fill dictionary with file_id as keys and file_name as values

for f in files:
    dl = r.get('{}bundle/{}/{}'.format(api, task_id, f),stream=True)                                 # Get a stream to the bundle file
    filename = os.path.basename(cgi.parse_header(dl.headers['Content-Disposition'])[1]['filename'])  # Parse name from Content-Disposition header 
    filepath = os.path.join(destDir, filename)                                                       # Create output file path
    with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
        for data in dl.iter_content(chunk_size=8192): f.write(data) 
print('Downloaded files can be found at: {}'.format(destDir))

