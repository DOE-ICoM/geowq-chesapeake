# make data/unique_pixeldays_w_bandvals.csv

import ee
import argparse

ee.Initialize()


def get_modis(date):
    filename = "modis-" + date.replace("-", "_")
    gdrive_folder = 'ICOM exports'
    asset = "users/jstacompute/icom_pixelcenters"
    dataset = 'MODIS/006/MYDOCGA'

    obs = ee.FeatureCollection(asset)
    ic = ee.ImageCollection(dataset)
    ic = ic.filterDate("2018-01-01", "2018-01-02")

    def getBandValues(im):
        bandValFC = ee.Algorithms.If(
            obs.size(
            ),  # If this is 0, we have no images in the collection and it evaluates to "False", else "True"
            im.reduceRegions(obs, ee.Reducer.first(),
                             scale=500),  #  True: grab its band values
            None  # False: only geometry and 'pixelday' properties will be set
        )

        return bandValFC

    bandVals = ic.map(getBandValues, opt_dropNulls=True).flatten()

    task = ee.batch.Export.table.toDrive(collection=bandVals,
                                         description=filename,
                                         fileFormat='CSV',
                                         folder=gdrive_folder)

    task.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    args = vars(parser.parse_args())
    date = args["date"]
    get_modis(date)


if __name__ == "__main__":
    main()