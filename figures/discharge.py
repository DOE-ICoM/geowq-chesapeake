import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/discharge_raw.csv", parse_dates=True, index_col="measurement_dt")
df = df[["discharge_va", "site_str"]]
df.sort_index(inplace=True)
df = df.groupby("site_str").rolling("365D").median().reset_index()
df = df[df["measurement_dt"] > pd.to_datetime("1945-01-01 00:00:00")]

fig, ax = plt.subplots()

g = sns.lineplot(data=df, x="measurement_dt", y="discharge_va", hue="site_str", ax=ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
# g.legend(handles,labels, ncol=3)
sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

g.set_yscale("log")
plt.xlabel("")
plt.ylabel("Discharge (cfs)")

# plt.show()
plt.savefig("figures/_discharge.pdf")
