##Sofia Avendano

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

##Grab data


def fit_sine(data):

    ##Read in Data
    #    data=pd.read_csv('temperature.csv')
    #    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].astype('datetime64')
    data['datetime'] = data['datetime'].dt.strftime('%j')
    data['datetime'] = pd.to_numeric(data["datetime"], downcast="float")
    date = data['datetime']
    #    diff_time=time_data[1:,]-time_data[:-1,]
    ##Get features
    temp = data['SST (C)'].values

    temp_doy = []
    for i in np.arange(1, 367):
        temp_mean = np.mean(data['SST (C)'][data['datetime'] == i])
        temp_doy.append(temp_mean)
    plt.scatter(np.arange(1, 367), temp_doy)
    plt.show()
    #    temp_doy=np.vstack((temp_doy, temp_doy))
    date_doy = np.arange(1, 367)
    #    date_doy=np.arange(0,366*2)

    ##https://scipy-cookbook.readthedocs.io/items/FittingData.html

    # Fit the first set
    #fitfunc = lambda p, x: p[0]*np.sin(2*np.pi/p[1]*x+p[2])+p[3] # Target function
    fitfunc = lambda p, x: p[0] * np.sin(2 * np.pi / p[1] * x + p[2]) + p[3]
    errfunc = lambda p, x, y: fitfunc(p, x
                                      ) - y  # Distance to the target function
    #    p0 = [1.21877004e+01, 3.15741655e+16, 2.95421427e+01, 1.65708992e+01] # Initial guess for the parameters
    p0 = [12.23414294, 368.06372131, 29.41529669, 16.38386361]
    #p1, success = optimize.leastsq(errfunc, p0[:], args=(date, temp))
    p1, success = optimize.leastsq(errfunc, p0[:], args=(date_doy, temp_doy))
    print(p1)
    print(success)
    #time = np.linspace(date.min(), date.max(), 1000)
    fitted_sine = fitfunc(p1, date)
    plt.figure(figsize=(20, 10))
    #    plt.plot(date, temp, "ro")
    #    plt.plot(date, fitfunc(p1, date), "bo") # Plot of the data and the fit

    plt.plot(date_doy, temp_doy, "ro")
    plt.plot(date_doy, fitfunc(p1, date_doy), "bo")
    plt.show()
    plt.savefig('fitted_sine.png')

    return fitted_sine
