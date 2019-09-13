import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger("pypsa")
logger.setLevel("WARNING")
import pypsa
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from n_dimensional_datasets import *
from plotter import *

# approximation methods
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import LinearNDInterpolator