##Sofia Avendano
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
import call_data2
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
## Analysis from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74



def build_grid():

    ## Analysis from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    #rf = RandomForestRegressor(random_state = 1)# Look at parameters used by our current forest
    #print('Parameters currently in use:\n')
    #pprint(rf.get_params())


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)]
    #n_estimators=1000

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']


    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)


    # Minimum number of samples required to split a node
    min_samples_split = [2, 6, 10]


    # Minimum number of samples required at each leaf node (floats would be considered fractions)
    min_samples_leaf = [ 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6]

    # Maximum number of leaf nodes
    max_leaf_nodes=[10, 20, 50, 100]
    max_leaf_nodes.append(None)

    # Method of selecting samples for training each tree (ie with or without resampling)
    bootstrap = [True, False]# Create the random grid



    ##Build Grid
    random_grid = {
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'max_leaf_nodes':max_leaf_nodes}
               
    pprint(random_grid)
    return random_grid




#max_leaf_nodes=None
#max_features: 3.18; sqrt
#n_estimators: 3.2; 1600
#max_depth: 3.10; 110
#min_samples_split: 3.06; 10
#min_samples_leaf: 4.42; 0.1; {'min_samples_leaf': 5}; 3.03
#bootstrap: 3.30; True
#max_leaf_nodes: 4.03; 10
#max_depth=90, max_features='sqrt', n_estimators=1000,



def tune_hyper_params(grid, params)
    #rf = RandomForestRegressor(bootstrap='True',max_depth=100, max_features='sqrt', min_samples_split=10, n_estimators=1600)
    rf = RandomForestRegressor(random_state=1)
    # Random search of parameters, using kfold cross validation
    # search across 100 different combinations, and use all available cores
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
    #rf_random = HalvingRandomSearchCV(estimator = rf, param_distributions = random_grid, cv = cv, verbose=2, random_state=1, n_jobs = -1, scoring='neg_mean_squared_error')# Fit the random search model
    rf_random = HalvingGridSearchCV(estimator = rf, param_grid = random_grid, cv = cv, verbose=2, random_state=1, n_jobs = -1, scoring='neg_mean_squared_error')# Fit the random search model

    #all_params=['datetime','Ratio 2', 'Ratio 3','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']
#    all_params=['datetime', 'longitude', 'latitude', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'sur_refl_b08', 'sur_refl_b12', 'sur_refl_b13', 'sur_refl_b16']

    X_train, y_train, X_test, y_test, feature_names = call_data2.clean_data('temperature','SST (C)', params)

    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    print(rf_random.best_score_)
    print(rf_random.best_estimator_)

    predictions=rf_random.predict(X_test)
    errors = abs(predictions - y_test)
    print(metrics.mean_squared_error(y_test, predictions))
    rmse = metrics.mean_squared_error(y_test, predictions
    return(rmse)
