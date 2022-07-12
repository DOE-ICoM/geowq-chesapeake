.PHONY: data

# ----

data: data_X_train

data_X_train: data/X_train_temperature.pkl data/X_train_salinity.pkl data/X_train_turbidity.pkl

data/X_train_temperature.pkl: scripts/01_rf_fit.py
	python $< --variable temperature --var_col "SST (C)"

data/X_train_salinity.pkl: scripts/01_rf_fit.py
	python $< --variable salinity --var_col "SSS (psu)"

data/X_train_turbidity.pkl: scripts/01_rf_fit.py
	python $< --variable turbidity --var_col "turbidity (NTU)"

# ----

data_rf_random: data/rf_random_temperature.pkl data/rf_random_salinity.pkl data/rf_random_turbidity.pkl

data/rf_random_temperature.pkl: scripts/02_rf_tune.py data/X_train_temperature.pkl
	python $< --variable temperature

data/rf_random_salinity.pkl: scripts/02_rf_tune.py data/X_train_salinity.pkl
	python $< --variable salinity

data/rf_random_turbidity.pkl: scripts/02_rf_tune.py data/X_train_turbidity.pkl
	python $< --variable turbidity

# ----

data/aggregated_w_bandvals.csv: src/run_satval.py
	python $<

# ----

figures: figures/00_combined.pdf

figures/00_combined.pdf: 
	pdftk $(wildcard figures/_*.pdf) output figures/00_combined.pdf
