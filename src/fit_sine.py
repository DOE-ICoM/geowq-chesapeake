# https://scipy-cookbook.readthedocs.io/items/FittingData.html

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt


def fit_sine(data):
    # data = pd.read_csv('temperature.csv')
    # data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # data = pd.read_csv("aggregated_w_bandvals.csv")

    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].astype('datetime64')
    data['datetime'] = data['datetime'].dt.strftime('%j')
    data['datetime'] = pd.to_numeric(data["datetime"], downcast="float")
    date = data['datetime']

    temp_doy = []
    for i in np.arange(1, 367):
        temp_mean = np.mean(data['SST (C)'][data['datetime'] == i])
        temp_doy.append(temp_mean)
    date_doy = np.arange(1, 367)

    fitfunc = lambda p, x: p[0] * np.sin(2 * np.pi / p[1] * x + p[2]) + p[3]
    errfunc = lambda p, x, y: fitfunc(p, x
                                      ) - y  # Distance to the target function
    p0 = [12.23414294, 368.06372131, 29.41529669, 16.38386361]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(date_doy, temp_doy))
    # print(p1)
    # print(success)
    fitted_sine = fitfunc(p1, date)
    
    plt.figure(figsize=(20, 10))
    plt.plot(date_doy, temp_doy, "ro")
    plt.plot(date_doy, fitfunc(p1, date_doy), "bo")
    plt.savefig('figures/fitted_sine.png')

    return fitted_sine
