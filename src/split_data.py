import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import csv
import itertools
from sklearn import preprocessing 

"""
Take all the data from the MODIS bands and filter for QC's all equal to 0
Then split the data into 3 .csv files for temperature, salinity and turbidity
Add in Ratios of spectral bands as in Geiger, 2013

"""

##Data
path = '/lclscratch/savendano/ICOM/'
data=pd.read_csv(path+'aggregated_w_bandvals.csv')


##adapted from run_satval.py

QCcols = [k for k in data.keys() if 'QC' in k] ##pick out QC keys

##Which variables are the labels?
variable = ['SSS (psu)','SST (C)','turbidity (NTU)']
filename=['salinity','temperature','turbidity']

##Look for spectral band data
sur =[k for k in data.keys() if 'sur_refl' in k]


for num, var in enumerate(variable):
    fn = filename[num]
    valid_data = [0] # which QC values are acceptable data
    dftemp = data[~pd.isna(data[var])]
    dftemp = dftemp.reset_index()

    sss_data=data[var][~pd.isna(data[var])]


    q_array = dftemp[QCcols].to_numpy()
    q_valid = np.zeros(q_array.shape)
    for v in valid_data:
        q_valid[q_array==v] = 1
    rowct_valid = q_valid.sum(axis=1)

    rowct_valid.shape
    print(rowct_valid)

  #  if fn==turbidity:
  #      for turbid in dftemp['turbidity (NTU)']:
  #          if turbid>600:



    data_filtered=[]
    for count,row in enumerate(rowct_valid):

        if row==9:
            data_filtered.append(count)

    dftemp=dftemp.loc[data_filtered]
#    for band in sur:
#        dftemp[band]=(dftemp[band]-dftemp[band].min())/(dftemp[band].max()-dftemp[band].min())
    ratio_1=dftemp['sur_refl_b08']/dftemp['sur_refl_b12']
    ratio_2=dftemp['sur_refl_b09']/dftemp['sur_refl_b12']
    ratio_3=dftemp['sur_refl_b10']/dftemp['sur_refl_b12']
    
    
    
    dftemp.insert(10,'Ratio 1',ratio_1)
    dftemp.insert(11,'Ratio 2',ratio_2)
    dftemp.insert(12,'Ratio 3',ratio_3)

    dftemp['Ratio 1']=dftemp['Ratio 1'].astype(np.float32)
    dftemp['Ratio 2']=dftemp['Ratio 2'].astype(np.float32)
    dftemp['Ratio 3']=dftemp['Ratio 3'].astype(np.float32)



    dftemp = dftemp[~pd.isnull(dftemp['Ratio 1'])]
    dftemp = dftemp[~pd.isnull(dftemp['Ratio 2'])]
    dftemp = dftemp[~pd.isnull(dftemp['Ratio 3'])]
    dftemp.fillna(0)
    dftemp=dftemp.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio 1', 'Ratio 2', 'Ratio 3'], how="all")
    dftemp = dftemp[~pd.isnull(dftemp['Ratio 1'])]
    dftemp = dftemp[~pd.isnull(dftemp['Ratio 2'])]
    dftemp = dftemp[~pd.isnull(dftemp['Ratio 3'])]


#    for col in QCcols:
#        sss_data.drop(col)
#    sss_data.drop(filename!=fn)
#    dftemp = pd.DataFrame(dftemp)
# save the dataframe as a csv file
    dftemp.to_csv(path+fn+'.csv')
