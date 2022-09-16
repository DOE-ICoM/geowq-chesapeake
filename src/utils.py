import os
import re
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt


def select_var(dt, var_col):
    if isinstance(dt, str):
        dt = pd.read_csv(dt)
    dt = dt.copy()

    # drop columns associated with all other vars from dt
    var_key = ["SST (C)", "depth (m)", "SSS (psu)", "turbidity (NTU)"]
    var_key.remove(var_col)

    dt = dt.loc[:, ~dt.columns.str.startswith(tuple(var_key))]
    dt.replace([np.inf, -np.inf], np.nan, inplace=True)

    return dt


def freq_count(dt, grp_cols):
    dt = dt.copy()
    dt = (dt.groupby(grp_cols).size().reset_index().rename(columns={
        0: "count"
    }).sort_values("count", ascending=False).reset_index(drop=True))
    return dt


def _unwrap(y):
    return [[x for x in re.split("\s+", y)]]


def load_md(variable, fpath_prestring="data/best_params_"):
    # load best_params_[variable].md
    bp_colnames = _unwrap(
        str(
            pd.read_table(fpath_prestring + variable + ".md",
                          nrows=1,
                          header=None)[0][0]).strip())[0]
    if "max_leaf_nodes" in bp_colnames:
        bp_colnames.remove("max_leaf_nodes")
    bp = pd.DataFrame(
        _unwrap(
            str(
                pd.read_table(fpath_prestring + variable + ".md",
                              skiprows=2,
                              header=None)[0][0]).strip()),
        columns=bp_colnames,
    )
    return bp


def clean_var_name(x):
    return "".join(map(str.lower, x)).split(" ")[0]


def datetime_to_doy(dt):
    dt = dt.copy()
    dt = pd.to_datetime(dt)
    ## Subtract from fitted sine wave/ Calculate day of Year instead
    #    data['datetime']=pd.to_datetime(data['datetime'], infer_datetime_format=True)
    dt = dt.astype("datetime64")
    dt = dt.dt.strftime("%j")
    dt = pd.to_numeric(dt, downcast="float")
    return dt


def tabulate_to_latex(tabulate_output, fname, caption='nasdf', table_num=1):
    with open(fname, "w") as f:
        f.write("\\documentclass{article}" + "\r\n")
    with open(fname, "a") as f:
        f.write("\\usepackage{standalone}" + "\r\n")
        f.write("\\usepackage[justification=centering]{caption}" + "\r\n")
        f.write("\\usepackage{adjustbox}" + "\r\n")
        f.write("\\setcounter{table}{" + str(table_num) + "}" + "\r\n")
        f.write("\\renewcommand\\thetable{\\arabic{table}}" + "\r\n")
        f.write("\\pagestyle{empty}" + "\r\n")
        f.write("\\begin{document}" + "\r\n")
        f.write("\\begin{table}" + "\r\n")
        f.write("\\caption{" + caption + "}" + "\r\n")
        f.write("\\begin{adjustbox}{center}" + "\r\n")
        f.write(tabulate_output + "\r\n")
        f.write("\\end{adjustbox}" + "\r\n")
        f.write("\\end{table}" + "\r\n")
        f.write("\\end{document}" + "\r\n")


def modisaqua_path(date, band="sur_refl_b08"):
    # date='2018-01-01'
    return "data/MODIS-Aqua/{date}_{band}.tif".format(date=date, band=band)


def get_rf_prediction(date):
    # date = "2022-09-04"

    bay_gdf_hires = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(
        epsg=4326)

    path_downsample = "data/prediction/" + date + "_downsample.tif"
    if not os.path.exists(path_downsample):
        call_string = "gdalwarp -te -77.3425000000000011 36.1675000000000040 -74.7974999999999994 39.6325000000000074 -ts 509 693 -overwrite data/prediction/" + date + ".tif data/prediction/" + date + "_downsample.tif"
        subprocess.call(call_string)

    img_rf = xr.open_dataset("data/prediction/" + date + "_downsample.tif",
                             engine="rasterio")
    img_rf = img_rf.rio.clip(bay_gdf_hires.geometry)
    img_rf = img_rf["band_data"].sel(band=1)
    img_rf.rio.to_raster("data/prediction/" + date + "_downsample_clip.tif")
    # img_rf.plot.imshow()
    # plt.show()
    return img_rf
