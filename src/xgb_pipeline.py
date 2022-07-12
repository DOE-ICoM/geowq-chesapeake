from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import call_data2
import scipy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from boruta import BorutaPy
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

##Parameters to Consider
print('call params')
all_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']


##Get Data
print('get data')
X_train, y_train, X_test, y_test, feature_names = call_data2.clean_data('salinity', 'SSS (psu)',all_params)

##Find RFE n features to select
min_features_to_select=5

#rfe_num.rfe_choose(X_train, y_train, all_params)
rfecv = RFECV(estimator=RandomForestRegressor(), step=1, scoring="neg_mean_squared_error", cv=5, verbose=2,min_features_to_select=min_features_to_select)

##Build Pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('feat_select', rfecv),
                 ('regressor', RandomForestRegressor())])


##Define parameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 20)]
n_estimators=[250,500,1000]

# Number of features to consider at every split
max_features = ['auto']


# Maximum number of levels in tree
max_depth = [10, 50, 100]
max_depth.append(None)


# Minimum number of samples required to split a node
min_samples_split = [.25,.5,.75,2,5]


# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 5, 15]


# Method of selecting samples for training each tree
bootstrap = [True, False]



##Build Search grid

random_grid = [{
              #{'selector':[RFE(estimator=RandomForestRegressor()),SelectKBest(score_func=mutual_info_regression,k=11),SelectFromModel(LogisticRegression(C=1, penalty='l2'))],
               'regressor':[RandomForestRegressor()],
               'regressor__n_estimators': n_estimators,
               'regressor__max_depth': max_depth,
               'regressor__min_samples_split': min_samples_split,
               'regressor__min_samples_leaf': min_samples_leaf,
               'regressor__bootstrap': bootstrap}]


##Fit parameters
#cv= RepeatedKFold(n_splits=3, n_repeats=5, random_state=1)
clf = RandomizedSearchCV(pipe, random_grid,n_iter=100, cv=5, verbose=1, scoring='neg_root_mean_squared_error',n_jobs=-1)
clf = clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_estimator_)
print(clf.best_score_)


#print("Optimal Num features: %d" % rfecv.n_features_)
#print(rfecv.support_)
#print(rfecv.ranking_)

#plt.figure(figsize=(12,6))
#plt.scatter(range(min_features_to_select,
#               len(rfecv.grid_scores_) + min_features_to_select),
#         rfecv.grid_scores_)
#plt.show()

##Print out most important features
#features = clf.named_steps['selector']

#print(X_train.columns[features.transform(np.arange(len(X_train.columns)))])

#support = clf.named_steps['selector'].support_
#feature_names = np.array(feature_names) # transformed list to array

#salinity=ff.rf_regress('salinity','SSS (psu)', ['Ratio 1', 'Ratio 2', 'Ratio 3','sur','latitude','longitude','SST (C)'],1000, 0.75, 10, 3, 1)

