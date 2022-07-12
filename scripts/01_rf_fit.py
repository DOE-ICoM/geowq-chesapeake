import sys
import pickle
import argparse

sys.path.append(".")
from src import rf_icom_utils as utils
from src import rf_icom_call_data2 as call_data2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="temperature", type=str)
    parser.add_argument("--var_col", default="SST (C)", type=str)
    args = vars(parser.parse_args())
    variable = args["variable"]
    var_col = args["var_col"]

    #variable='salinity'
    #var_col='SSS (psu)'

    ##Suggested Parameters for Salinity and Temperature
    ##note that temperature will include a fitted sine curve as a feature as well
    predictors = [
        'datetime', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'sur_refl_b08',
        'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11', 'sur_refl_b12',
        'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15', 'sur_refl_b16',
        'latitude', 'longitude'
    ]

    ##Split into training and test data, test size=.33
    print('split data')
    X_train, y_train, X_test, y_test, feature_names = call_data2.clean_data(
        variable, var_col, predictors, test_size=0.33)

    ##Feature Selection vs Hyperparameter Tuning: https://stats.stackexchange.com/questions/264533/how-should-feature-selection-and-hyperparameter-optimization-be-ordered-in-the-m
    ##I added n_estimators=1000, and a max_depth =20 for the hyperparameters used in feature selection

    #Run Feature Selection
    print('run feature selection')
    X_train, X_test = utils.run_rfe(X_train, y_train, X_test, y_test,
                                    feature_names, variable)

    pickle.dump(X_train, open("data/X_train_" + variable + ".pkl", "wb"))
    pickle.dump(X_test, open("data/X_test_" + variable + ".pkl", "wb"))
    pickle.dump(y_train, open("data/y_train_" + variable + ".pkl", "wb"))
    pickle.dump(y_test, open("data/y_test_" + variable + ".pkl", "wb"))


if __name__ == "__main__":
    main()
