.PHONY: data

# ----

data: data/X_train.pkl data/rf_random.pkl

data/aggregated_w_bandvals.csv: src/run_satval.py
	python $<

data/X_train.pkl: scripts/01_rf_fit.py
	python $<

data/rf_random.pkl: scripts/02_rf_tune.py
	python $<

# ----

figures: figures/00_combined.pdf

figures/00_combined.pdf: 
	pdftk $(wildcard figures/_*.pdf) output figures/00_combined.pdf
