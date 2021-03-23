import pandas as pd
import numpy as np
import csv
from sklearn import linear_model

## read data
data = pd.read_csv("train.csv")
data = data.to_numpy()
y = data[:,1]
X = data[:,2:7]

# transform data
X_transformed = np.concatenate((X, 
                                np.power(X, 2), 
                                np.exp(X), 
                                np.cos(X), 
                                np.ones([len(X), 1])), axis=1)

# model setup
model = linear_model.Ridge(alpha=0.01, fit_intercept=False)
reg = model.fit(X_transformed, y)

# output
np.savetxt('sub.csv', reg.coef_, newline='\n')