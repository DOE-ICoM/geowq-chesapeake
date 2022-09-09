import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(".")
from src import rf_icom_call_data2 as call_data2


# scatter plot of measured vs predicted
variable = "salinity"
var_col = "SSS (psu)"

rf_random_path = "data/rf_random_" + variable + ".pkl"
rf_random = pickle.load(open(rf_random_path, "rb"))

predictors = pickle.load(open("data/imp_params_" + variable + ".pkl", "rb"))

X_test = pickle.load(open("data/X_test_" + variable + ".pkl", "rb"))
y_test = pickle.load(open("data/y_test_" + variable + ".pkl", "rb"))

predictions = rf_random.predict(X_test)
res = pd.DataFrame(y_test, columns=["obs"])
res["predict"] = predictions.copy()

plt.close()
fig, ax = plt.subplots(figsize=(8, 4))

g = sns.scatterplot(data=res, x="predict", y="obs", ax=ax, s=7, color=".15")
# increasing bin makes coarser boxes, increasing pthresh makes less boxes
sns.histplot(data=res, x="predict", y="obs", bins=100, pthresh=0.2,cmap="mako", ax=ax)
plt.plot([0, 28], [0, 28], color="red")
ax.set_xlim(-0.5, 32)
ax.set_ylim(-0.5, 32)
g.set_xlabel("Predicted")
g.set_ylabel("Observed")

X = res["predict"].to_numpy().reshape(-1,1)
y = res["obs"].to_numpy()
reg = LinearRegression().fit(X,y)

eq = "y = " + str(round(reg.coef_[0], 2)) + "x - " + str(round(abs(reg.intercept_), 2))
ax.text(3, 27, eq)
ax.text(3, 24, "$R^2$ = " + str(round(reg.score(X, y), 2)))

# plt.show()
plt.savefig("figures/_validation.pdf")