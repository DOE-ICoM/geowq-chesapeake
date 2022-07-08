##Sofia Avendano
import utils
import call_data2



##Suggested Parameters for Salinity and Temperature
##note that temperature will include a fitted sine curve as a feature as well
all_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']

#variable='salinity'
#var_col='SSS (psu)'


variable = 'temperature'
var_col='SST (C)'

##Split into training and test data, test size=.33
print('split data')
X_train, y_train, X_test, y_test, feature_names=call_data2.clean_data(variable, var_col,all_params,test_size=0.33)




##Feature Selection vs Hyperparameter Tuning: https://stats.stackexchange.com/questions/264533/how-should-feature-selection-and-hyperparameter-optimization-be-ordered-in-the-m
##I added n_estimators=1000, and a max_depth =20 for the hyperparameters used in feature selection 

#Run Feature Selection
print('run feature selection')

X_train,X_test=utils.run_rfe(X_train, y_train, X_test, y_test,feature_names)

#Get randomm grid
print('random grid')
random_grid=utils.build_grid()

##Tune hyperparameters
print('tune hyperparameters')

rmse, best_params = utils.tune_hyper_params(random_grid, all_params, X_train, y_train, X_test, y_test)
print('Final RMSE:')
print(rmse)
print('Best-fit Parameters')
print(best_params)



