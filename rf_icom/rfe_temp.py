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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import metrics

##Parameters to Consider
all_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','sur_refl_b08','sur_refl_b09','sur_refl_b10','sur_refl_b11','sur_refl_b12','sur_refl_b13','sur_refl_b14','sur_refl_b15','sur_refl_b16','latitude','longitude']


X, y, X_test, y_test, feature_names = call_data2.clean_data('temperature', 'SST (C)',all_params)

n_scores_mean=[]
n_scores_std=[]
ols = RandomForestRegressor(n_estimators=250, max_depth=20, random_state=1)
min_features_to_select=5
rfecv = RFECV(estimator=ols, step=1, scoring="neg_root_mean_squared_error", cv=5, verbose=2,min_features_to_select=min_features_to_select)

rfecv.fit(X, y)

X=rfecv.transform(X)
X_test=rfecv.transform(X_test)


print(rfecv)
print("Optimal Num features: %d" % rfecv.n_features_)
print(rfecv.support_)
print(rfecv.ranking_)
print(rfecv.estimator_.feature_importances_)
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

fig=plt.figure
plt.bar(important_params, rfecv.estimator_.feature_importances_)
plt.show()


#important_params=['datetime','Ratio 1','Ratio 2', 'Ratio 3','SST (C)','sur_refl_b09','sur_refl_b14','sur_refl_b15','longitude']

#X, y, X_test, y_test, feature_names = call_data.clean_data('temperature', 'SST (C)',important_params)

#ols = RandomForestRegressor(max_samples=0.5,n_estimators=1000,random_state=1)
ols.fit(X, y)
predictions=ols.predict(X_test)
errors = abs(predictions - y_test)
print(metrics.mean_squared_error(y_test, predictions, squared=False))


