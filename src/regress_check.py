from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import csv
import itertools
import matplotlib.pyplot as plt


##Read in Data


def rf_regress(variable, var_col, predictors, estimator, sample, split, repeat, state): 

    

    

    ##Read in Data

    data=pd.read_csv(variable+'.csv')    

    y = data[var_col].values # target

    time_format='%m/%d/%Y %H:%M'

#    data['datetime']=pd.to_datetime(data['datetime'], infer_datetime_format=True)

    data['datetime'] = data['datetime'].astype('datetime64').astype(int).astype(float)

    ##Find Predictors

    notsur=[]

    for s in data.keys():

        for predictor in predictors:        

            if predictor in s:

                break 

    

            elif s in notsur:

                break

            elif s in predictors:

                break

            else:

                notsur.append(s)


    print(notsur)


    #notsur = [s for s in data.keys() if 'sur' not in s]

    X = data.drop(notsur, axis = 1).values

    print('X')

    print(X)    

    
    X, X_test, y, y_test=train_test_split(X, y,test_size=0.5)
    

    ##Choose Model

    # define the model

    model = RandomForestRegressor(n_estimators=estimator, max_samples=sample)

    # evaluate the model

    cv = RepeatedKFold(n_splits=split, n_repeats=repeat, random_state=state)

    n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

    # report performance

    rms = np.sqrt(np.mean(n_scores**2))

    mae=mean(n_scores)

    print(variable)

    print(rms)

    model.fit(X,y)

    importance = model.feature_importances_

    print(importance)

    return rms


##Call for different variables

salinity=rf_regress('salinity','SSS (psu)', ['Ratio 1', 'Ratio 2', 'Ratio 3','sur','latitude','longitude','SST (C)'],100, 0.75, 10, 3, 1)

salinity=rf_regress('salinity','SSS (psu)', ['Ratio 1', 'Ratio 2', 'Ratio 3','sur','latitude','longitude','SST (C)'],1000, 0.75, 10, 3, 1)
print(salinity)



