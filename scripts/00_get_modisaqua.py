import ee
import sys
import requests

sys.path.append("src")
from src import utils

ee.Initialize()

startDate = '2018-01-01'
endDate = '2018-01-02'
band = "sur_refl_b08"

# bottom-left, top-right
region = ee.Geometry.Rectangle(-77.576, 36.661, -75.605, 39.632)

collection = ee.ImageCollection('MODIS/006/MYDOCGA').filterBounds(
    region).filterDate(startDate, endDate)
composite = collection.median().select(band)

# test = composite.getInfo()
# test["bands"][1]["id"]

url = composite.getDownloadURL(
            {
                "format": "GEO_TIFF",
                "filename": "test.tif",
                "region": region,
                "scale": 200,
            }
        )


out_path = utils.modisaqua_path(startDate, band)
response = requests.get(url)
with open(out_path, "wb") as fd:
        fd.write(response.content)

# task = ee.batch.Export.image.toDrive(image=composite,
#                               description='image',
#                               scale=200,
#                               region=region)

# task.start()
