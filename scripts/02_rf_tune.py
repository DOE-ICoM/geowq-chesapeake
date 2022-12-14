import sys
import pickle
import warnings
import argparse
import numpy as np
from tabulate import tabulate

sys.path.append(".")
from src import rf_icom_utils as utils

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="temperature", type=str)
    args = vars(parser.parse_args())
    variable = args["variable"]

    X_train = pickle.load(open("data/X_train_" + variable + ".pkl", "rb"))
    X_test = pickle.load(open("data/X_test_" + variable + ".pkl", "rb"))
    y_train = pickle.load(open("data/y_train_" + variable + ".pkl", "rb"))
    y_test = pickle.load(open("data/y_test_" + variable + ".pkl", "rb"))

    #Get random grid
    print('random grid')
    random_grid = utils.build_grid()

    ##Tune hyperparameters
    print('tune hyperparameters')
    rmse, best_params = utils.tune_hyper_params(random_grid,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                variable,
                                                overwrite=True)

    print('Final RMSE:')
    print(rmse)
    if variable == "turbidity":
        rmse = np.exp(rmse)
    with open("data/rmse_" + variable + ".md", 'w') as f:
        f.write(tabulate([[rmse]], headers=["rmse"]))

    print('Best-fit Parameters')
    print(best_params)
    with open("data/best_params_" + variable + ".md", 'w') as f:
        f.write(
            tabulate([[i for i in best_params.values()]],
                     headers=[k for k in best_params.keys()]))


if __name__ == "__main__":
    main()