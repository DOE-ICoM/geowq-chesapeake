import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df3 = pd.read_csv("data/discharge_raw.csv")
df3["measurement_dt"] = pd.to_datetime(df3["measurement_dt"])

fig, ax = plt.subplots()

g = sns.lineplot(data=df3,
                 x="measurement_dt",
                 y="discharge_va",
                 hue="site_str", 
                 ax=ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])

g.set_yscale("log")

plt.xlabel("")
plt.ylabel("Discharge (cfs)")

# plt.show()
plt.savefig("figures/_discharge.pdf")