##Sofia Avendano

gradient boosting for turbidity? Works better for more complicated problems. 
Turbdity did much better with log-transform.

What files are in the rf_icom directory:


---------
Getting Started:

split_data.py: this splits the data into 3 different .csv files (turbidity, salinity and temperature) based on QC values

Use split_data.py once before  the analysis

----------
Running the model:

utils.py: has feature selection, grid construction and hyperparameter tuning functions

call_data2.py: cleans up the data in the selected .csv file; for temperature calls fit_sine.py to  fit a sine curve to the data

call_script.py: calls utils.py and call_data2.py to tune and run the full random forest model

---------
Other Scripts:

pipeline.py: creates a pipeline with rfecv and hyperparameter tuning; was not able to get good results using this method  but  probably ultimately want to end up using a pipeline instead
A pipeline makes sure that you don't overfit--that the same training data are used throughout.

fit_sine.py: fits the temperature data to a sine curve
