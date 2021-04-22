##Sofia Avendano

import numpy as  np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd



##Grab data

def fit_sine(data):


    ##Read in Data
#    data=pd.read_csv('temperature.csv')
#    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['datetime']=pd.to_datetime(data['datetime'])
#    data['datetime'] = data['datetime'].astype('datetime64')
#    data['datetime']=data['datetime'].dt.strftime('%j')
    data['datetime']=pd.to_numeric(data["datetime"], downcast="float")
    date=data['datetime']
#    diff_time=time_data[1:,]-time_data[:-1,]
    ##Get features
    temp = data['SST (C)'].values


    #plt.scatter(date, temp)
    #plt.show()


    ##https://scipy-cookbook.readthedocs.io/items/FittingData.html

    # Fit the first set
    fitfunc = lambda p, x: p[0]*np.sin(2*np.pi/p[1]*x+p[2])+p[3] # Target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
    p0 = [1.21877004e+01, 3.15741655e+16, 2.95421427e+01, 1.65708992e+01] # Initial guess for the parameters
    #p0=[33, 180, 2.95421427e+01]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(date, temp))
    print(p1)
    print(success)
    #time = np.linspace(date.min(), date.max(), 1000)
    fitted_sine=fitfunc(p1,date)
    plt.figure(figsize=(20,10))
    plt.plot(date, temp, "ro")
    plt.plot(date, fitfunc(p1, date), "bo") # Plot of the data and the fit
    plt.show()

    return fitted_sine


