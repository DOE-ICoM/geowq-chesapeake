import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(".")
from src import utils
from src import fit_sine


def _get_predictions(variable):
    # variable = "temperature"
    rf_random_path = "data/rf_random_" + variable + ".pkl"
    rf_random = pickle.load(open(rf_random_path, "rb"))

    X_test = pickle.load(open("data/X_test_" + variable + ".pkl", "rb"))
    y_test = pickle.load(open("data/y_test_" + variable + ".pkl", "rb"))

    predictions = rf_random.predict(X_test)
    res = pd.DataFrame(y_test, columns=["obs"])
    res["predict"] = predictions.copy()

    if variable == "temperature":
        # back out non-sine adjusted values
        p1 = pickle.load(open("data/temperature_sine_coef.pkl", "rb"))
        offset = fit_sine.fitfunc(p1, X_test[0:, 0])
        res["predict"] = res["predict"].copy() + offset
        res["obs"] = res["obs"].copy() + offset

    return res


variables = ["SSS (psu)", "turbidity (NTU)", "SST (C)"]
variables_str = ["salinity (psu)", "turbidity (NTU)", "temperature (C)"]
variables_str_short = [utils.clean_var_name(v) for v in variables_str]

# scatter plot of measured vs predicted
res = [_get_predictions(variable) for variable in variables_str_short]


def _plot(ax,
          dt,
          title,
          xy_min=None,
          xy_max=None,
          bins=100,
          pthresh=0.2,
          hatching=True,
          text_anchor=(3, 27),
          text_space=2,
          ylab=False):
    # dt = res[1]
    if xy_max is None:
        xy_max = max(max(dt["obs"]), max(dt["predict"]))
        print(xy_max)
    if xy_min is None:
        xy_min = min(min(dt["obs"]), min(dt["predict"]))
        print(xy_min)

    g = sns.scatterplot(data=dt, x="predict", y="obs", ax=ax, s=7, color=".15")
    # increasing bin makes coarser boxes, increasing pthresh makes less boxes
    if hatching:
        sns.histplot(data=dt,
                     x="predict",
                     y="obs",
                     bins=bins,
                     pthresh=pthresh,
                     color='black',
                     hue_order="black",
                     pmax=0,
                     ax=ax)
    ax.plot([0, xy_max - 4], [0, xy_max - 4], color="red")
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    g.set_xlabel("Predicted")
    if ylab:
        g.set_ylabel("Observed")
    else:
        g.set_ylabel("")

    X = dt["predict"].to_numpy().reshape(-1, 1)
    y = dt["obs"].to_numpy()
    reg = LinearRegression().fit(X, y)

    eq = "y = " + str(round(reg.coef_[0], 2)) + "x - " + str(
        round(abs(reg.intercept_), 2))
    r2 = str(round(reg.score(X, y), 2))
    ax.text(text_anchor[0], text_anchor[1], eq, size=8)
    ax.text(text_anchor[0],
            text_anchor[1] - text_space,
            "$R^2$ = " + r2,
            size=8)
    ax.set_title(title)
    ax.set_aspect("equal")

    return ax, r2

# # sine-adjusted temperature plot
# plt.close()
# fig, ax = plt.subplots(figsize=(8, 4))
# _plot(ax, res[2], title="asdf", xy_max=10, xy_min=-10)
# plt.show()

plt.close()
r2 = []
fig, axes = plt.subplots(figsize=(8, 4.5), ncols=3)
r2.append(_plot(axes[0], res[0], "salinity", -0.5, 32, ylab=True, text_space=2.5)[1])
r2.append(_plot(axes[1],
      res[1],
      "turbidity",
      xy_max=75,
      xy_min=0,
      hatching=False,
      text_anchor=(37, 10),
      text_space=5)[1])
r2.append(_plot(axes[2],
      res[2],
      "temperature",
      xy_min=-2,
      xy_max=35,
      text_anchor=(3, 30),
      text_space=2.5)[1])
# plt.show()

r2_df = pd.DataFrame(r2, columns=["r2"])
r2_df["variable"] = variables_str_short
r2_df.to_csv("data/r2.csv", index=False)

plt.savefig("figures/_validation.pdf")