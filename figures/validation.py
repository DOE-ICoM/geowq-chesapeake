import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(".")
from src import rf_icom_call_data2 as call_data2


# scatter plot of measured vs predicted
variable = "salinity"
var_col = "SSS (psu)"

rf_random_path = "data/rf_random_" + variable + ".pkl"
rf_random = pickle.load(open(rf_random_path, "rb"))

predictors = pickle.load(open("data/imp_params_" + variable + ".pkl", "rb"))

X_predict_raw = pd.read_csv("data/data_w_fwi.csv").dropna()
X_predict = call_data2.clean_data(variable,
                                  var_col,
                                  predictors,
                                  test_size=0,
                                  data=X_predict_raw)

predictions = rf_random.predict(X_predict[0])
res = pd.DataFrame(X_predict[1], columns=["obs"])
res["predict"] = predictions.copy()

plt.close()
fig, ax = plt.subplots(figsize=(8, 4))

g = sns.scatterplot(data=res, x="predict", y="obs")
plt.plot([0, 28], [0, 28], color="red")

ax.set_xlim(-0.5, 32)
ax.set_ylim(-0.5, 32)
g.set_xlabel("Predicted")
g.set_xlabel("Observed")

# plt.show()
plt.savefig("figures/_validation.pdf")