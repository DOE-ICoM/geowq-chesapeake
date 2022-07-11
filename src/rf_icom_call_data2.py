import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src import fit_sine


def clean_data(variable, var_col, predictors, test_size=0.5):
    """
    Get the data from the split .csv files based on the label, remove
    unwanted columns; remove NaNs; split temperature into fitted sine and
    variablity of temperature from the fitted sine if the label is 
    temperature

    Inputs:
    variable => string, name of .csv file and label
    var_col => string, name of label column
    test_size => optional, default splits 50% training, 50% testing
    predictors => list of strings with all the wanted feature names

    Output:
    feature_names => list, names of feature columns
    X_train, X_test => array, features for train and test sets
    y_train, y_test => array, label for train and test sets
    
    """

    ##Read in Data
    # data = pd.read_csv(variable + '.csv')
    data = pd.read_csv("data/aggregated_w_bandvals.csv")
    var_key = ["SST (C)", "depth (m)", "SSS (psu)", "turbidity (NTU)"]
    var_key.remove(var_col)
    data = data.loc[:, ~data.columns.str.startswith(tuple(var_key))]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    ##For turbidity get rid of negative numbers
    print(len(data))
    if var_col == 'turbidity (NTU)':
        data = data[data['turbidity (NTU)'] > 0]

    print(len(data))

    ##Find Predictors and Variable Column
    notsur = []
    for s in data.keys():
        for predictor in predictors:
            if predictor in s:
                break

            elif s in notsur:
                break
            elif s in predictors:
                break
            elif s == var_col:
                break

            else:
                notsur.append(s)

    ##Get rid of unnecessary data
    data = data.drop(notsur, axis=1)
    data = data.dropna()

    data['datetime'] = pd.to_datetime(data['datetime'])
    #    data['datetime'] = data['datetime'].astype('datetime64')
    #    data['datetime']=data['datetime'].dt.strftime('%j')
    data['datetime'] = pd.to_numeric(data["datetime"], downcast="float")

    if variable == 'temperature':
        print(data['SST (C)'])
        fitted_sine = fit_sine.fit_sine(data)
        # print('fitted_sine')
        # print(fitted_sine)
        data['SST (C)'] = data['SST (C)'] - fitted_sine
        data.insert(3, "fitted_sine", fitted_sine)
        predictors.append('fitted_sine')

    #    plt.scatter(data['datetime'],data['SST (C)'])
    #    plt.show()

    ##Convert to Day of Year
    data['datetime'] = pd.to_datetime(data['datetime'])
    ##Subtract from fitted sine wave/ Calculate day of Year instead
    #    data['datetime']=pd.to_datetime(data['datetime'], infer_datetime_format=True)
    data['datetime'] = data['datetime'].astype('datetime64')
    #.astype(int).astype(float)
    data['datetime'] = data['datetime'].dt.strftime('%j')
    data['datetime'] = pd.to_numeric(data["datetime"], downcast="float")

    #    print(len(data))
    #    data=data.drop(data['turbidity (NTU)']<0)
    #    print(len(data))

    ##Get features
    y = data[var_col].values
    X = data.drop(var_col, axis=1).values
    feature_names = [k for k in data.keys() if k in predictors]

    plt.figure(figsize=(20, 10))
    data.hist()
    plt.tight_layout()
    plt.savefig("figures/" + variable + '_data_hist.png')

    ##split everything

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size)

    return X_train, y_train, X_test, y_test, feature_names
