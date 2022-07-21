.PHONY: data

variables := sss sst turbidity

test:
	@echo $(variables_parsed)


ICOM_DATA = D:/ICOM
export ICOM_DATA:=$(ICOM_DATA)

env:
	python -c "import os; print(os.environ['ICOM_DATA'])"

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

# all_data.csv is not created with this code base...

# data filtered to be within CB and a surface obs in the modis time-window
data_filtered: $(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/filtered.csv

$(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/filtered.csv: scripts/00_get_data.py
	python $< --target filtered.csv	

pixel_centers_shp: $(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/pixel_centers.shp

$(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/pixel_centers.shp: scripts/00_get_data.py \
	| data_filtered
	python $< --target pixel_centers.shp

data_aggregated_csv: $(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/aggregated.csv

$(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/aggregated.csv: scripts/00_get_data.py
	python $< --target aggregated.csv

data_aggregated_gee_csv: $(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/aggregated_gee.csv

$(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/aggregated_gee.csv: scripts/00_get_data.py
	python $< --target aggregated_gee.csv

data/unique_pixeldays_w_bandvals.csv: scripts/00_get_data.py data_aggregated_gee_csv
	python $< --target unique_pixeldays_w_bandvals

data_aggregated_w_bandvals_csv: $(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/aggregated_w_bandvals.csv

$(ICOM_DATA)/Modeling\ Data/Processed\ Data\ p1/aggregated_w_bandvals.csv: scripts/00_get_data.py
	python $< --target aggregated_w_bandvals.csv

# ----

figures: figures/00_combined.pdf

figures/_obs_stats.pdf: figures/obs_stats.py
	python $<
	pdftk $(wildcard figures/obs_*.pdf) output $@

path_map_counts := $(addprefix figures/, $(addsuffix _map_counts.pdf, ${variables}))

figures/_map_counts_all.pdf: $(path_map_counts)
	pdftk $(wildcard figures/*_map_counts.pdf) output figures/_map_counts_all.pdf

figures/%_map_counts.pdf: figures/plot_data.py figures/plot_helpers.py
	python $<

path_map_variable := $(addprefix figures/, $(addsuffix _map_variable.pdf, ${variables}))

figures/_map_variable_all.pdf: $(path_map_variable)
	pdftk $(wildcard figures/*_map_variable.pdf) output figures/_map_variable_all.pdf

figures/%_map_variable.pdf: figures/plot_data.py figures/plot_helpers.py
	python $<

figures/00_combined.pdf: figures/_obs_stats.pdf \
	figures/_map_counts_all.pdf figures/_map_variable_all.pdf 
	pdftk $(wildcard figures/_*.pdf) output figures/00_combined.pdf
