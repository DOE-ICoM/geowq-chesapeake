import pandas as pd
import seaborn as sns
import dataretrieval.nwis as nwis
import matplotlib.pyplot as plt

nwis_stations = {
    "Nanticoke": "01488110",
    "Choptank": "01491000",
    "Susquehanna": "01576000",
    "Potomac": "01654000",
    "Pautexent": "01594440"
}

stations = [x for x in nwis_stations.values()]

df3 = nwis.get_discharge_measurements(sites=stations)[0]
df3["measurement_dt"] = pd.to_datetime(df3["measurement_dt"])
df3 = df3.merge(pd.DataFrame(nwis_stations,
                             index=[0]).T.reset_index().rename(columns={
                                 "index": "site_str",
                                 0: "site_no"
                             }),
                on="site_no")

# df3.columns
# df3["site_str"].unique()

g = sns.lineplot(data=df3, x="measurement_dt", y="discharge_va", hue="site_str")
g.set_yscale("log")
plt.show()
