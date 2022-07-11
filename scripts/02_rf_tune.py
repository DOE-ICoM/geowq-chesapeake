import sys
import pickle
import warnings

sys.path.append(".")
from src import rf_icom_utils as utils

warnings.simplefilter(action='ignore', category=FutureWarning)

predictors = [
    'datetime', 'Ratio 1', 'Ratio 2', 'Ratio 3', 'sur_refl_b08',
    'sur_refl_b09', 'sur_refl_b10', 'sur_refl_b11', 'sur_refl_b12',
    'sur_refl_b13', 'sur_refl_b14', 'sur_refl_b15', 'sur_refl_b16', 'latitude',
    'longitude'
]

X_train = pickle.load(open("data/X_train.pkl", "rb"))
X_test = pickle.load(open("data/X_test.pkl", "rb"))
y_train = pickle.load(open("data/y_train.pkl", "rb"))
y_test = pickle.load(open("data/y_test.pkl", "rb"))

#Get random grid
print('random grid')
random_grid = utils.build_grid()

##Tune hyperparameters
print('tune hyperparameters')

rmse, best_params = utils.tune_hyper_params(random_grid, predictors, X_train,
                                            y_train, X_test, y_test)
print('Final RMSE:')
print(rmse)
print('Best-fit Parameters')
print(best_params)

# TODO: save rmse and best_params
