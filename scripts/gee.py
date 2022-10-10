import pickle
import numpy as np
import pandas as pd
# import tensorflow_decision_forests as tfdf


def split_dataset(dataset, test_ratio=0.30):  
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

output_bucket = "rabpro-gee-uploads"
model_dir = "gs://" + output_bucket + "/rf_model"

variable = "temperature"
X_train = pickle.load(open("data/X_train_" + variable + ".pkl", "rb"))
X_test = pickle.load(open("data/X_test_" + variable + ".pkl", "rb"))
y_train = pickle.load(open("data/y_train_" + variable + ".pkl", "rb"))
y_test = pickle.load(open("data/y_test_" + variable + ".pkl", "rb"))
important_params = pickle.load(open("data/imp_params_" + variable + ".pkl", "rb"))

train_ds_pd = pd.DataFrame(X_train, columns=important_params)
train_ds_pd[variable] = y_train

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=variable, task=tfdf.keras.Task.REGRESSION)

model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model.compile(metrics=["mse"])
model.fit(x=train_ds)

model.save("rf_model", save_format="tf")
# model.save(model_dir, save_format="tf")