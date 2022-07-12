# geowq

[![Paper DOI](https://img.shields.io/badge/Paper-XXXX/XXXX-blue.svg)](https://doi.org) [![Code DOI](https://img.shields.io/badge/Code-XXXX/XXXX-blue.svg)](https://doi.org) [![Data DOI](https://img.shields.io/badge/Data-XXXX/XXXX-blue.svg)](https://doi.org)

Geographically aware estimates of remotely sensed water properties for Chesapeake Bay

## Setup

```shell
conda env create -f environment.yml
```

## Usage

### Generate data

```shell
python scripts/00_get_data.py
```

### RF model

```shell
# variable selection + initial fitting
python scripts/01_rf_fit.py

# hyperparameter tuning
python scripts/02_rf_tune.py
```
