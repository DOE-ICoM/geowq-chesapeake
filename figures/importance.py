import pickle
import pandas as pd
import matplotlib.pyplot as plt

variable_list = ["salinity", "temperature", "turbidity"]

def imp_plot(variable):
    # variable = "temperature"
    print(variable)
    rfecv_path = "data/rfecv_" + variable + ".pkl"
    rfecv = pickle.load(open(rfecv_path, "rb"))
    feature_names = pickle.load(open("data/imp_params_" + variable + ".pkl", "rb"))

    dset = pd.DataFrame()
    dset["attr"] = feature_names
    dset["importance"] = rfecv.estimator_.feature_importances_
    dset = dset.sort_values(by="importance", ascending=False)
    dset.loc[dset["attr"] == "cost", "attr"] = "fwi"

    plt.figure(figsize=(16, 14))

    plt.barh(y=dset["attr"], width=dset["importance"], color="#1976D2")
    plt.title(variable + " - Feature Importances", fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Importance", fontsize=14, labelpad=20)

    # plt.show()
    plt.savefig("figures/" + variable + "_importance.pdf", bbox_inches='tight')

[imp_plot(variable) for variable in variable_list]
