##Sofia Avendano
import rf
import call_data
import feat_select
from sklearn.model_selection import train_test_split

##missing depth
all_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','SST (C)','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']

params_no_ratios=['SST (C)','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']


variable='salinity'
var_col='SSS (psu)'

##Split into training and test data, test size=.33
X_train, y_train, X_test, y_test, feature_names=call_data.data(variable, var_col,all_params,test_size=0.33)


##Split into hyper and feature selection sets
X_hyper, X_feat, y_hyper, y_feat=train_test_split(X_train, y_train,test_size=0.5)


##Run feature selection
feature_ranks=feat_select.rf_boruta(X_feat, y_feat, feature_names, variable, var_col, all_params, depth=5)

##Run hyperparameters

##Run full model


