##Sofia Avendano
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
import call_data2
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score


def run_rfe(X_train, y_train, X_test, y_test, feature_names):
    """
    Run recursive feature extraction using RFECV
    Inputs: 
    X_train => array of feature values (training set)
    y_train => array of label values (training set)
    X_test => array of feature values (test set)
    y_test => array of label values (test set)
    feature_names => list of feature column names
    Outputs: 
    X_train and X_test with non important features culled

    """
    print(type(X_train))
    print(type(y_train))
    print(type(feature_names))

    X = X_train
    y = y_train
    n_scores_mean = []
    n_scores_std = []
    ols = RandomForestRegressor(n_estimators=250, max_depth=20, random_state=1)
    min_features_to_select = 5
    rfecv = RFECV(estimator=ols,
                  step=1,
                  scoring="neg_root_mean_squared_error",
                  cv=5,
                  verbose=2,
                  min_features_to_select=min_features_to_select)

    rfecv.fit(X, y)

    X = rfecv.transform(X)
    X_test = rfecv.transform(X_test)

    print(rfecv)
    print("Optimal Num features: %d" % rfecv.n_features_)
    print(rfecv.support_)
    print(rfecv.ranking_)
    print(rfecv.estimator_.feature_importances_)
    important_params = []
    idx = 0
    for i in rfecv.ranking_:
        if i == 1:
            important_params.append(feature_names[idx])
        else:
            pass
        idx = idx + 1
    print(important_params)

    plt.figure
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean Squared error")
    plt.scatter(
        range(min_features_to_select,
              len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_)
    plt.show()
    plt.savefig('rfecv.png')

    plt.figure(figsize=(15, 10))
    plt.bar(important_params, rfecv.estimator_.feature_importances_)
    #  plt.xticks(important_params, rfecv.estimator_.feature_importances_, rotation=90)
    plt.show()
    plt.savefig('important_features.png')

    ols.fit(X, y)
    predictions = ols.predict(X_test)
    errors = abs(predictions - y_test)
    print(metrics.mean_squared_error(y_test, predictions))
    return X, X_test


def build_grid():
    """
    Builds hyperparameter grid. 
    Probably not the best way to do this, but can edit function to change grid

    Outputs: hyperparameter dictionary

    """
    ## Analysis from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=11)]
    #n_estimators=1000

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 6, 10]

    # Minimum number of samples required at each leaf node (floats would be considered fractions)
    min_samples_leaf = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6]

    # Maximum number of leaf nodes
    max_leaf_nodes = [10, 20, 50, 100]
    max_leaf_nodes.append(None)

    # Method of selecting samples for training each tree (ie with or without resampling)
    bootstrap = [True, False]  # Create the random grid

    ##Build Grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'max_leaf_nodes': max_leaf_nodes
    }

    pprint(random_grid)
    return random_grid


def tune_hyper_params(grid, params, X_train, y_train, X_test, y_test):
    """
    Runs Halving Random Search to compare different hyperparamter combinations
    Inputs: 
    parameter grid (can use build_grid())
    parameters (found by feature selection)
    X_train and X_test (can use X arrays from run_rfe)
    y_train and y_test
    
    Outputs: 
    MSE of best run
    best run parameters

    """
    #leave regressor as simple as possible so just testing against defaults
    rf = RandomForestRegressor(random_state=1)

    # Random search of parameters, using kfold cross validation
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)

    rf_random = HalvingRandomSearchCV(
        estimator=rf,
        param_distributions=grid,
        cv=cv,
        verbose=2,
        random_state=1,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error')  # Fit the random search model

    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    print(rf_random.best_score_)
    print(rf_random.best_estimator_)

    predictions = rf_random.predict(X_test)
    errors = abs(predictions - y_test)
    print(metrics.mean_squared_error(y_test, predictions))
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse, rf_random.best_params_
