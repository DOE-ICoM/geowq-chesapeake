import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df3 = pd.read_csv("data/discharge_raw.csv")
df3["measurement_dt"] = pd.to_datetime(df3["measurement_dt"])

g = sns.lineplot(data=df3,
                 x="measurement_dt",
                 y="discharge_va",
                 hue="site_str")
g.set_yscale("log")
plt.savefig("figures/discharge.pdf")