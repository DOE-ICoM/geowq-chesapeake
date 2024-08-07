# make data/unique_pixeldays_w_bandvals.csv

import ee
import requests
import argparse
from datetime import datetime
from datetime import timedelta

ee.Initialize()


def get_modis(date):
    # date = "2022-09-04"
    filename = "modis-" + date.replace("-", "_")
    gdrive_folder = 'ICOM exports'
    asset = "users/jstacompute/icom_pixelcenters_4326"
    dataset = 'MODIS/006/MYDOCGA'

    obs = ee.FeatureCollection(asset)
    ic = ee.ImageCollection(dataset)
    date_plus_one = str(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).split(" ")[0]
    ic = ic.filterDate(date, date_plus_one)

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