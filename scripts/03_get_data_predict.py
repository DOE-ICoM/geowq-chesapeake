# make data/unique_pixeldays_w_bandvals.csv

import ee
import requests
import argparse
import pandas as pd

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

    return (bandVals.getDownloadURL(filetype="csv",
                                    filename=filename), filename)


def _fetch_data(url, fname="test.csv"):
    response = requests.get(url)
    with open("data/prediction/" + fname + ".csv", "wb") as fd:
        fd.write(response.content)

    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    args = vars(parser.parse_args())
    date = args["date"]
    (url, filename) = get_modis(date)
    data = _fetch_data(url, filename)
    # pd.read_csv(data).head()


if __name__ == "__main__":
    main()