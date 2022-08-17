# geowq

[![Paper DOI](https://img.shields.io/badge/Paper-XXXX/XXXX-blue.svg)](https://doi.org) [![Code DOI](https://img.shields.io/badge/Code-XXXX/XXXX-blue.svg)](https://doi.org) [![Data DOI](https://img.shields.io/badge/Data-XXXX/XXXX-blue.svg)](https://doi.org)

Geographically aware estimates of remotely sensed water properties for Chesapeake Bay

## Setup

```shell
# software
conda env create -f environment.yml
```

<!-- ```shell
# observtional data
createdb -U postgres icom
set PGPASSWORD=password psql -U postgres -d icom -c 'CREATE EXTENSION postgis;'
python scripts/00_load_all_data.py
``` -->

## Usage

### Generate training data

```shell
# executes scripts/00_get_data.py
make data_aggregated_gee_csv
# manually upload aggregated_gee.csv to GEE
# manually download results to data/unique_pixeldays_w_bandvals.csv
make data_aggregated_w_bandvals_csv

# add freshwater influence predictor
make data_w_fwi
```

### Train RF model

```shell
# variable selection + initial fitting
make data/X_train_salinity.pkl

# hyperparameter tuning
make data/rf_random_salinity.pkl
```

### Pull prediction data

```shell
python scripts/03_get_data_predict.py --date "2018-01-02"
```

### Generate prediction surfaces

```shell
python scripts/04_rf_predict.py --date "2018-01-02"
```
