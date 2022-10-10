

var theMap = ui.Map();
ui.root.clear();

colDatesPanel = function () {
    var startDayLabel = ui.Label('Date:');
    var startDayBox = ui.Textbox({ value: '2022-06-10' });
    startDayBox.style().set('stretch', 'horizontal');

    return ui.Panel(
        [
            ui.Panel(
                [startDayLabel, startDayBox],
                ui.Panel.Layout.Flow('horizontal'), { stretch: 'horizontal' }
            )
        ]
    );
};
visSelectPanel = function () {
    var indexList = ['Salinity', 'Temperature', 'Turbidity'];
    var indexSelect = ui.Select({ items: indexList, value: 'Salinity', style: { stretch: 'horizontal' } });
    return ui.Panel([indexSelect], null, { stretch: 'horizontal' });
};

var region = ee.Geometry.Rectangle(-77.576, 36.661, -75.605, 39.632);

var startDate = '2018-01-01'
var endDate = '2018-01-02'

var collection = ee.ImageCollection('MODIS/006/MYDOCGA')
    .filterDate(startDate, endDate)
    .map(function (image) { return image.clip(region) });

theMap.centerObject(region, 8)
theMap.addLayer(collection)

var controlPanel = ui.Panel({
    style: { width: '350px', position: 'top-left' } //, backgroundColor: 'rgba(255, 255, 255, 0)'
});

var colDatesPanel = colDatesPanel();
var visSelectPanel = visSelectPanel();

var url = ui.Label({
    value: 'About geowq',
});
url.setUrl('https://github.com/DOE-ICoM/geowq');

controlPanel.add(colDatesPanel);
controlPanel.add(visSelectPanel);
controlPanel.add(url);

ui.root.add(controlPanel);
ui.root.add(theMap);
