##Sofia Avendano
##from https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/


# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import call_data2
from sklearn import metrics




##Parameters to Consider
all_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']

##Get Data
X_train, y_train, X_test, y_test, feature_names = call_data2.clean_data('salinity', 'SSS (psu)',all_params, test_size=0.33)
print(y_train)
print(type(y_train[0]))
# fit model no training data
model = XGBRegressor()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#errors = abs(predictions - y_test)
print(metrics.mean_squared_error(y_test, predictions, squared=False))
## evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))




