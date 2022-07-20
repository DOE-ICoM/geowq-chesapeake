import os
import pandas as pd

from figures import plot_helpers as ph

columns = ['SST (C)', 'SSS (psu)', 'turbidity (NTU)']
startDate = '2002-01-01 00:00:00'
endDate = '2021-01-01 00:00:00'


def _import_data():
    filepath = os.environ["icom_data"] + "/Modeling Data/Processed Data p1/"
    filename = "aggregated.csv"
    df = pd.read_csv(filepath + filename)
    df = ph.split_pixelday(df)  # add 'pix_id' and 'date' columns
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.index = df['datetime']
    return df


df = _import_data()

# --- temporal plots
# number of modis pixels containing an insitu observation each day
[ph.plot_counts(df, column) for column in columns]

# mean time series of an individual or group of modis pixels
pix_ids = [274463844, 270230367]
[ph.plot_timeseries(df, pix_ids, column) for column in columns]

# --- spatial plots
# map the number of days containing at least 1 valid observation in each modis pixel
[
    ph.map_counts(df, column, startDate=startDate, endDate=endDate, num=100)
    for column in columns
]

# map the mean column value over a given date range.
[
    ph.map_variable(df, column, startDate=startDate, endDate=endDate)
    for column in columns
]
