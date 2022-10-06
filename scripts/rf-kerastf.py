import sys
import pickle

sys.path.append(".")
from src import scikit_learn_model_converter as sci2tf

variable = "salinity"

path = "data/rf_random_" + variable + ".pkl"
# path = "data/rfecv_" + variable + ".pkl"
# path = "data/ols_" + variable + ".pkl"

model = pickle.load(open(path, "rb"))

test = model.best_estimator_

tensorflow_model = sci2tf.convert(test)

type(tensorflow_model)
dir(tensorflow_model)
tensorflow_model.save("model", save_format="tf")
