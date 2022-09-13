import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(".")
from src import rf_icom_call_data2 as call_data2
from src import utils


def _get_predictions(variable):
    rf_random_path = "data/rf_random_" + variable + ".pkl"
    rf_random = pickle.load(open(rf_random_path, "rb"))

    X_test = pickle.load(open("data/X_test_" + variable + ".pkl", "rb"))
    y_test = pickle.load(open("data/y_test_" + variable + ".pkl", "rb"))

    predictions = rf_random.predict(X_test)
    res = pd.DataFrame(y_test, columns=["obs"])
    res["predict"] = predictions.copy()
    return res


variables = ["SSS (psu)", "turbidity (NTU)", "SST (C)"]
variables_str = ["salinity (psu)", "turbidity (NTU)", "temperature (C)"]
variables_str_short = [utils.clean_var_name(v) for v in variables_str]

# scatter plot of measured vs predicted
res = [_get_predictions(variable) for variable in variables_str_short]


def _plot(ax,
          dt,
          xy_min=None,
          xy_max=None,
          bins=100,
          pthresh=0.2,
          hatching=True):
    # dt = res[1]
    if xy_max is None:
        xy_max = max(max(dt["obs"]), max(dt["predict"]))
    if xy_min is None:
        xy_min = min(min(dt["obs"]), min(dt["predict"]))

    g = sns.scatterplot(data=dt, x="predict", y="obs", ax=ax, s=7, color=".15")
    # increasing bin makes coarser boxes, increasing pthresh makes less boxes
    if hatching:
        sns.histplot(data=dt,
                     x="predict",
                     y="obs",
                     bins=bins,
                     pthresh=pthresh,
                     cmap="mako",
                     ax=ax)
    plt.plot([0, xy_max - 4], [0, xy_max - 4], color="red")
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    g.set_xlabel("Predicted")
    g.set_ylabel("Observed")

    X = dt["predict"].to_numpy().reshape(-1, 1)
    y = dt["obs"].to_numpy()
    reg = LinearRegression().fit(X, y)

    eq = "y = " + str(round(reg.coef_[0], 2)) + "x - " + str(
        round(abs(reg.intercept_), 2))
    ax.text(3, 27, eq)
    ax.text(3, 24, "$R^2$ = " + str(round(reg.score(X, y), 2)))
    return ax


plt.close()
fig, ax = plt.subplots(figsize=(8, 4))
_plot(ax, res[0], -0.5, 32)
# _plot(ax, res[1], xy_max=75, hatching=False)
# _plot(ax, res[2], xy_min=-10, xy_max=32)
# plt.show()

plt.savefig("figures/_validation.pdf")