import sys
import pickle
import tensorflow as tf

sys.path.append(".")
from src import scikit_learn_model_converter as sci2tf

variable = "salinity"

path = "data/rf_random_" + variable + ".pkl"
# path = "data/rfecv_" + variable + ".pkl"
# path = "data/ols_" + variable + ".pkl"

model = pickle.load(open(path, "rb"))
X_test = pickle.load(open("data/X_test_" + variable + ".pkl", "rb"))
y_test = pickle.load(open("data/y_test_" + variable + ".pkl", "rb"))
X_train = pickle.load(open("data/X_train_" + variable + ".pkl", "rb"))
y_train = pickle.load(open("data/y_train_" + variable + ".pkl", "rb"))
imp_params = pickle.load(open("data/imp_params_" + variable + ".pkl", "rb"))

test = pd.DataFrame(X_test, columns=imp_params)
test["y"] = y_test
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test, label="y", task=tfdf.keras.Task.REGRESSION)

train = pd.DataFrame(X_train, columns=imp_params)
train["y"] = y_train
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label="y", task=tfdf.keras.Task.REGRESSION)

tensorflow_model = sci2tf.convert(model.best_estimator_)

tensorflow_model.compile()
tensorflow_model.fit(x=train_ds)
tensorflow_model.evaluate(test_ds)

type(tensorflow_model)
dir(tensorflow_model)
tensorflow_model.save("model", save_format="tf")


# https://www.tensorflow.org/decision_forests/tutorials/beginner_colab#training_a_regression_model

import math
import requests
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


# url = "https://storage.googleapis.com/download.tensorflow.org/data/abalone_raw.csv"
# out_path = "abalone_raw.csv"
# response = requests.get(url)
# with open(out_path, "wb") as fd:
#     fd.write(response.content)

dataset_df = pd.read_csv("abalone_raw.csv")
print(dataset_df.head(3))

# Split the dataset into a training and testing dataset.
train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

# Name of the label column.
label = "Rings"

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

# Configure the model.
model_7 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)

# Train the model.
model_7.fit(x=train_ds)

model_7.compile(metrics=["mse"])
evaluation = model_7.evaluate(test_ds, return_dict=True)

print(evaluation)
print()
print(f"MSE: {evaluation['mse']}")
print(f"RMSE: {math.sqrt(evaluation['mse'])}")

