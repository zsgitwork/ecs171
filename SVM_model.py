### SVM MODEL ###

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm  # library for SVM

# load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#display head of csv file (just to visualize)
print(df.head())

# define X (data) and y (target)  #FIXME: not working rn bc data is in df
X = df.data
y = df.target

#intializing model
model = svm.SVC()