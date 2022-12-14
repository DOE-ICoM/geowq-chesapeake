import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/discharge_raw.csv", parse_dates=True, index_col="measurement_dt")
df = df[["discharge_va", "site_str"]]
df = df[df.index > pd.to_datetime("1945-01-01 00:00:00")]
df.sort_index(inplace=True)
df = df.groupby("site_str").rolling("365D").median().reset_index()

fig, ax = plt.subplots()

g = sns.lineplot(
    data=df, x="measurement_dt", y="discharge_va", hue="site_str", ax=ax, markers=True
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.legend(loc=(0.05, -0.2), ncol=3)

g.set_yscale("log")
plt.xlabel("")
plt.ylabel("Discharge (cfs)")

# plt.show()
plt.savefig("figures/_discharge.pdf", bbox_inches='tight')
