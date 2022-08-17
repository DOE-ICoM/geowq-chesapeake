import os
import pandas as pd
import dataretrieval.nwis as nwis

nwis_stations = {
    "Nanticoke": "01488110",
    "Choptank": "01491000",
    "Susquehanna": "01576000",
    "Potomac": "01654000",
    "Pautexent": "01594440",
    "James": "02037500"
}

stations = [x for x in nwis_stations.values()]

outpath = "data/discharge_raw.csv"
if not os.path.exists(outpath):
    df3 = nwis.get_discharge_measurements(sites=stations)[0]
    df3["measurement_dt"] = pd.to_datetime(df3["measurement_dt"])
    df3 = df3.merge(pd.DataFrame(nwis_stations,
                                 index=[0]).T.reset_index().rename(columns={
                                     "index": "site_str",
                                     0: "site_no"
                                 }),
                    on="site_no")
    df3.to_csv(outpath, index=False)

df3 = pd.read_csv(outpath)
df3["measurement_dt"] = pd.to_datetime(df3["measurement_dt"])

# df3.columns
# df3["site_str"].unique()

pd.DataFrame(
    df3.groupby("site_str")["discharge_va"].median()).reset_index().to_csv(
        "data/discharge_median.csv", index=False)
