import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

print(tf.__version__)

date = "2022-09-04"

dataset = 'MODIS/006/MYDOCGA'
asset_pixelcenters = "users/jstacompute/icom_pixelcenters_4326"

# load static asset