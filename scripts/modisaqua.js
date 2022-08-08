// bottom-left, top-right
var region = ee.Geometry.Rectangle(-77.576, 36.661, -75.605, 39.632);

var startDate = '2018-01-01'
var endDate = '2018-01-02'

var collection = ee.ImageCollection('MODIS/006/MYDOCGA')
    .filterBounds(region)
    .filterDate(startDate, endDate)

// var composite = collection.median().select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']);

// Visualize the composite
Map.centerObject(region, 8)
Map.addLayer(collection)

// Export.image.toDrive({
//   image: composite,
//   description: 'image',
//   scale: 200,
//   region: region
// });