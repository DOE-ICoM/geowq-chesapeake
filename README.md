# geowq

[![Paper DOI](https://img.shields.io/badge/Paper-XXXX/XXXX-blue.svg)](https://doi.org) [![Code DOI](https://img.shields.io/badge/Code-XXXX/XXXX-blue.svg)](https://doi.org) [![Data DOI](https://img.shields.io/badge/Data-XXXX/XXXX-blue.svg)](https://doi.org)

Geographically aware approaches for remotely sensed estimates of estuarine water properties

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
python scripts/01_rf_fit.py
python scripts/02_rf_tune.py
```
