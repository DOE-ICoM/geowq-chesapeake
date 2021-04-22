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
import boruta
import call_data

##Boruta analysis from Aaron Lee (https://towardsdatascience.com/simple-example-using-boruta-feature-selection-in-python-8b96925d5d7a)

def rf_boruta(X, y, feature_names, variable, var_col, params, depth): 
##remove multicollinearity and missing values
    model = RandomForestRegressor(n_estimators=1000,max_depth=depth)

    ##grab training set

    model.fit(X,y)

    feat_selector = boruta.BorutaPy(model, verbose=1, random_state=1, n_estimators=1000)
 
    feat_selector.fit(X, y)

    feat_selector.support_

    feat_selector.ranking_

    X_filtered = feat_selector.transform(X)

    feature_ranks = list(zip(feature_names, 
                         feat_selector.ranking_, 
                         feat_selector.support_))
    print(feature_ranks)    
    
    
    return feature_ranks

##Call for different variables
#salinity=rf_regress('salinity','SSS (psu)', ['sur','latitude','longitude'],100, 0.5, 10, 3, 1)
#turbidity=rf_regress('turbidity','turbidity (NTU)', ['sur','datetime','latitude','longitude','depth'],100, 0.9,100, 3, 1)
#temperature=rf_regress('temperature','SST (C)',['sur', 'latitude','longitude','datetime'],100, 0.5, 10, 3, 1)
##salinity_model=rf_regress('salinity','SSS (psu)', ['Ratio 1','SST (C)','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude'],100, 0.5, 10, 3, 1)
