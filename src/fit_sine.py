# https://scipy-cookbook.readthedocs.io/items/FittingData.html

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt

fitfunc = lambda p, x: p[0] * np.sin(2 * np.pi / p[1] * x + p[2]) + p[3]


def fit_sine(data):
    # data = pd.read_csv("data/aggregated_w_bandvals.csv")

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

    errfunc = lambda p, x, y: fitfunc(p, x
                                      ) - y  # Distance to the target function
    p0 = [12.23414294, 368.06372131, 29.41529669, 16.38386361]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(date_doy, temp_doy))
    # print(success)
    # print(p1)
    pickle.dump(p1, open("data/temperature_sine_coef.pkl", "wb"))

    fitted_sine = fitfunc(p1, date)

    plt.rc('axes', labelsize=14)

    fig = plt.figure(figsize=[3.4/1.4, 2.8/1.4])
    ax = fig.add_subplot(111)
    ax.plot(date_doy, temp_doy, "ro")
    ax.plot(date_doy, fitfunc(p1, date_doy), "bo", markersize=2)    
    plt.xlabel("DOY", fontweight="bold")    
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.xaxis.set_ticks_position('bottom')
    # plt.show()
    plt.tight_layout()
    labels = ax.get_xticklabels()
    [label.set_fontweight('bold') for label in labels]
    plt.savefig('figures/fitted_sine.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(date_doy, temp_doy, "ro")
    ax.plot(date_doy, fitfunc(p1, date_doy), "bo", markersize=2)
    plt.xlabel("DOY", fontweight="bold")
    plt.tight_layout()
    plt.savefig('figures/fitted_sine_detail.png')
    
    # data_fitted = pd.DataFrame({"datetime":date_doy, "fit":fitfunc(p1, date_doy)})
    # test = data.merge(data_fitted)
    # test["diff"] = test["SST (C)"] - test["fit"]
    # fig, ax = plt.subplots()
    # sns.scatterplot(x="datetime", y="SST (C)", data=test, ax=ax)
    # sns.lineplot(x="datetime", y="fit", data=test, ax=ax)
    # plt.show()
    # sns.scatterplot(x="datetime", y="diff", data = test)
    # plt.show()

    return fitted_sine
