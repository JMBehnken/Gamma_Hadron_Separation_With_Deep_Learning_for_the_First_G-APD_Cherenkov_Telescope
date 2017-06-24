import pandas as pd
import numpy as np
import random
import json
import gzip
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
#from sklearn.metrics import auc

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense


import matplotlib.pyplot as plt
from itertools import islice
import itertools