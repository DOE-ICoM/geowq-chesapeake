import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(".")
from src import satval_class as sc


def filtered():
    cd = sc.satval(r"data\\params.csv")
    # cd.paths['filtered']
    cd.get_points()
    cd.assign_unique_location_ids()


def pixel_centers():
    cd = sc.satval(r"data\\params.csv")
    cd.map_coordinates_to_pixels()


def aggregated():
    cd = sc.satval(r"data\\params.csv")
    cd.aggregate_data_to_unique_pixeldays()


def aggregated_gee():
    cd = sc.satval(r"data\\params.csv")
    cd.start_gee_bandval_retrieval()


def unique_pixeldays_w_bandvals():
    cd = sc.satval(r"data\\params.csv")
    cd.start_gee_bandval_retrieval("users/jstacompute/icom_gee")


def aggregated_w_bandvals():
    cd = sc.satval(r"data\\params.csv")
    cd.merge_bandvals_and_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    args = vars(parser.parse_args())
    target = args["target"]

    methods = {
        'filtered.csv': filtered,
        "pixel_centers.shp": pixel_centers,
        "aggregated.csv": aggregated,
        "aggregated_gee.csv": aggregated_gee,
        "unique_pixeldays_w_bandvals": unique_pixeldays_w_bandvals,
        "aggregated_w_bandvals.csv": aggregated_w_bandvals
    }
    methods[target]()


# ---

# cd.map_coordinates_to_pixels()
# cd.aggregate_data_to_unique_pixeldays()

# # Try to start a GEE task by directly uploading the geodataframe - this
# # will fail for large datasets (e.g. > 20,000 rows)
# cd.start_gee_bandval_retrieval()
# # # Failed, so we upload the shapefile manually and pass in the asset location
# # gee_asset = 'users/jonschwenk/aggregated_gee_p1'
# # cd.start_gee_bandval_retrieval(asset=gee_asset)
# """Need to wait for GEE task to finish and download the .csv"""
# path_bandvals = r"C:\Users\Jon\Desktop\Research\ICoM\Data\Processed Data p1\unique_pixeldays_w_bandvals.csv"
# cd.merge_bandvals_and_data(path_bandvals)

# ---

# Compute data availability for specific variables and considering QC bands
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt

# QCcols = [k for k in cd.aggregated.keys() if 'QC' in k]
# var = 'SSS (psu)'
# var = 'SST (C)'
# var = 'turbidity (NTU)'

# valid_data = [0]  # which QC values are acceptable data
# dftemp = cd.aggregated[~pd.isna(cd.aggregated[var])]
# q_array = dftemp[QCcols].to_numpy()
# q_valid = np.zeros(q_array.shape)
# for v in valid_data:
#     q_valid[q_array == v] = 1
# rowct_valid = q_valid.sum(axis=1)
# all_valid_ct = np.sum(rowct_valid == 9)
# plt.close()
# plt.hist(rowct_valid, bins=np.arange(0, 11) - 0.5, width=0.9)
# plt.xlabel('# good bands for an observation')
# plt.ylabel('count')
# plt.title(var)

if __name__ == "__main__":
    main()