##Sofia Avendano

import numpy
import pandas as pd
import itertools
import boruta
import rf


####All possible variables: 'sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude','depth (m)','day_of_year'



def params():

    salinity_model, rmse, X, y=rf.rf_regress('salinity','SSS (psu)', ['sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude','depth (m)','day_of_year'],100, 0.5, 10, 3, 1)
    
    # define Boruta feature selection method
    feat_selector = boruta.BorutaPy(salinity_model, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    
    return X_filtered


X_filtered=params()


