import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src import utils
from src import fit_sine


def clean_data(variable, var_col, predictors, test_size=0.5, data=None):
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

    Examples:
    ```python
    from src import rf_icom_call_data2 as call_data2
    predictors = [
        'datetime', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'sur_refl_b08',
        'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11', 'sur_refl_b12',
        'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15', 'sur_refl_b16',
        "cost", "latitude", "longitude"
    ]
    X_train, y_train, X_test, y_test, feature_names = call_data2.clean_data(
        "temperature", "SST (C)", predictors
    )
    ```
    """    

    if data is None:
        data = pd.read_csv("data/aggregated_w_bandvals.csv")

    data = utils.select_var(data, var_col)

    print(len(data))
    
    ## For turbidity, get rid of negative numbers
    if test_size > 0: # aka we are not in prediction mode    
        if var_col == "turbidity (NTU)":
            data = data[data["turbidity (NTU)"] > 0]

    ## For turbidity, log scale data
    if var_col == "turbidity (NTU)":            
            data["turbidity (NTU)"] = np.log(data["turbidity (NTU)"])

    ## For salinity, set negative numbers to 0
    if var_col == "SSS (psu)":        
        data.loc[data["SSS (psu)"] < 0, "SSS (psu)"] = 0        

    print(len(data))

    ## Find Predictors and Variable Column
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

    ## Get rid of unnecessary data
    data = data.drop(notsur, axis=1)
    data = data.dropna()

    data["datetime"] = pd.to_datetime(data["datetime"])
    #    data['datetime'] = data['datetime'].astype('datetime64')
    #    data['datetime']=data['datetime'].dt.strftime('%j')
    data["datetime"] = pd.to_numeric(data["datetime"], downcast="float")

    if variable == "temperature":
        fitted_sine = fit_sine.fit_sine(data)
        # print('fitted_sine')
        # print(fitted_sine)
        data["SST (C)"] = data["SST (C)"] - fitted_sine
        data.insert(3, "fitted_sine", fitted_sine)
        predictors.append("fitted_sine")

    #    plt.scatter(data['datetime'],data['SST (C)'])
    #    plt.show()

    ## Convert to Day of Year
    # data["datetime"] = utils.datetime_to_doy(data["datetime"])
    # ummm, the data is already in doy?

    ## Get features
    y = data[var_col].values
    X = data.drop(var_col, axis=1)
    print(X.columns)
    lon_list = [lon for lon in X.longitude.values]
    lat_list = [lon for lon in X.latitude.values]

    print(X.shape)
    X = X.values
    feature_names = [k for k in data.keys() if k in predictors]

    plt.figure(figsize=(20, 10))
    try:
        data.hist()
    except:        
        data.drop(columns=["datetime"]).hist()
    plt.tight_layout()
    plt.savefig("figures/" + variable + "_data_hist.png")

    if test_size == 0:
        return X, y, lon_list, lat_list

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, y_train, X_test, y_test, feature_names
