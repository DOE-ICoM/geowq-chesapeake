## Geographically aware estimates of remotely sensed water properties for Chesapeake Bay

[![Paper DOI](https://img.shields.io/badge/Paper-XXXX/XXXX-blue.svg)](https://doi.org) [![Code DOI](https://img.shields.io/badge/Code-10.5281/zenodo.7332558-blue.svg)](https://doi.org/10.5281/zenodo.7332558) [![Data DOI](https://img.shields.io/badge/Data-10.6084/m9.figshare.21578898-blue.svg)](https://doi.org/10.6084/m9.figshare.21578898)

### Products

* ****[Supplementary figures and tables](https://github.com/DOE-ICoM/geowq/blob/master/manuscript/supplement.pdf)****

<!-- * [Google Earth Engine App](https://jstacompute.users.earthengine.app/view/geowq) -->

### Setup

#### Software

```shell
conda env create -f environment.yml
```

#### Data

* Locate the observational data file `all_data.csv` (download from figshare if necessary)

* Sync its path with the location defined in `data/params.csv`

### Usage

#### Generate training data

```shell
# executes scripts/00_get_data.py
make data_aggregated_gee_csv
# manually upload aggregated_gee.csv to GEE
# manually download results to data/unique_pixeldays_w_bandvals.csv
make data_aggregated_w_bandvals_csv

# add freshwater influence predictor
make data_w_fwi
```

#### Train RF model

```shell
# variable selection + initial fitting
make data/X_train_salinity.pkl

# hyperparameter tuning
make data/rf_random_salinity.pkl
```

#### Pull prediction data

```shell
python scripts/03_get_data_predict.py --date "2022-09-04"
```

#### Generate prediction surfaces

```shell
python scripts/04_rf_predict.py --date "2022-09-04" --variable salinity --var_col "SSS (psu)"
python scripts/04_rf_predict.py --date "2022-09-04" --variable turbidity --var_col "turbidity (NTU)"
python scripts/04_rf_predict.py --date "2022-09-04" --variable temperature --var_col "SST (C)"
```

### Release

This software has been approved for open source release and has been assigned identifier **C23002**.

### Copyright

© 2022. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
