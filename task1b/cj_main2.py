import pandas as pd
import numpy as np
import csv
import statistics
from sklearn import linear_model
from sklearn.model_selection import cross_validate

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

alphas = np.logspace(-4, 3, 100)
avg_scores_lasso = []
avg_scores_ridge = []

for alpha in alphas:
    lassomodel = linear_model.Lasso(alpha = alpha, fit_intercept=False)
    scores = cross_validate(lassomodel, X, y, cv = 10, scoring ='neg_root_mean_squared_error')
    avg_scores_lasso.append(statistics.mean(-scores['test_score']))
print(alphas[np.argmin(avg_scores_lasso)])
print(min(avg_scores_lasso))

for alpha in alphas:
    ridgemodel = linear_model.Ridge(alpha = alpha, fit_intercept=False)
    scores = cross_validate(ridgemodel, X, y, cv = 10, scoring ='neg_root_mean_squared_error')
    avg_scores_ridge.append(statistics.mean(-scores['test_score']))
print(alphas[np.argmin(avg_scores_ridge)])
print(min(avg_scores_ridge))


# model setup
model = linear_model.Ridge(alpha=alphas[np.argmin(avg_scores_lasso)], fit_intercept=False)
reg = model.fit(X_transformed, y)

np.savetxt('sub.csv', reg.coef_, newline='\n')