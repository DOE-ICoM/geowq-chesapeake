import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import call_data2
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import preprocessing
from sklearn import metrics

##Parameters to Consider
all_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','SSS (psu)','SST (C)','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']


X, y, X_test, y_test, feature_names = call_data2.clean_data('turbidity', 'turbidity (NTU)',all_params)

#nrmlzd = preprocessing.StandardScaler()
#X = nrmlzd.fit_transform(X)
#nrmlzd_y=preprocessing.StandardScaler()
#y = nrmlzd_y.fit_transform(y.reshape(-1,1))
#y = y.ravel()

#mean_y=nrmlzd_y.mean_
#std_y=nrmlzd_y.scale_
#print(y)

#y=y.reshape(-1,1)

#scaler=preprocessing.StandardScaler().fit(y)
#scaler.mean_
#scaler.scale_
#y_scaled=scaler.transform(y)
#y_scaled=y_scaled.ravel()



#y_test=y_test.reshape(-1,1)
#scaler_test=preprocessing.StandardScaler().fit(y_test)
#scaler_test.mean_
#scaler_test.scale_
#y_test_scaled=scaler_test.transform(y_test)
#y_test_scaled=y_test_scaled.ravel()







n_scores_mean=[]
n_scores_std=[]
ols = RandomForestRegressor(n_estimators=250,max_depth=20,random_state=1)
min_features_to_select=5
rfecv = RFECV(estimator=ols, step=1, scoring="neg_root_mean_squared_error", cv=5, verbose=2,min_features_to_select=min_features_to_select)

rfecv.fit(X, y)

print(len(X))

print(rfecv)
print("Optimal Num features: %d" % rfecv.n_features_)
print(rfecv.support_)
print(rfecv.ranking_)

important_params=[]
idx=0
for i in rfecv.ranking_:
     if  i == 1:
         important_params.append(feature_names[idx])
     else:
         pass
     idx=idx+1
print(important_params)


#rfecv_df = pd.DataFrame(rfecv.ranking_,index=list(X).columns,columns=['Rank']).sort_values(by='Rank',ascending=True)

#rfecv_df.head()
plt.figure
plt.xlabel("Number of features selected")
plt.ylabel("Mean Squared error")
plt.scatter(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()
print(rfecv.grid_scores_)
print(min_features_to_select)
plt.savefig('rfecv_turbidity')

ols.fit(X, y)
predictions=ols.predict(X_test)
errors = abs(predictions - y_test)
print(metrics.mean_squared_error(y_test, predictions, squared=False))


#dset = pd.DataFrame()
#dset['attr'] = feature_names
#dset['importance'] = rfecv.estimator_.feature_importances_

#dset = dset.sort_values(by='importance', ascending=False)


#plt.figure(figsize=(16, 14))
#plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
#plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
#plt.xlabel('Importance', fontsize=14, labelpad=20)
#plt.show()


#important_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','SST (C)','sur_refl_b09','sur_refl_b14','sur_refl_b15','longitude']

#X, y, X_test, y_test, feature_names = call_data.clean_data('salinity', 'SSS (psu)',important_params)

#model = RandomForestRegressor(n_estimators=250,max_depth=20,random_state=1)
#print('Parameters currently in use:\n')
#print(model.get_params())

#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, error_score='raise')

#n_scores_mean.append(np.mean(n_scores))
#n_scores_std.append(np.std(n_scores))
#rmse_rescaled=-np.mean(n_scores)*std_y+mean_y
#print('RMSE: %.3f (%.3f)' % (-np.mean(n_scores), np.std(n_scores)))

#print('Test Set')
#n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, error_score='raise')

#n_scores_mean.append(np.mean(n_scores))
#n_scores_std.append(np.std(n_scores))

#print('RMSE: %.3f (%.3f)' % (-np.mean(n_scores), np.std(n_scores)))
