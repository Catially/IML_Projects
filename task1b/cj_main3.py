import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


l1_ratios = np.linspace(0, 1, 10)
alphas = np.linspace(0, 1, 10)
error = np.zeros((len(l1_ratios), len(alphas)))
min_error = 0
best_alpha = 0
best_l1_ratios = 0

for i in range(len(l1_ratios)):
    for j in range(len(alphas)):
        r = linear_model.ElasticNet(alpha=alphas[j], l1_ratio=l1_ratios[i], fit_intercept=False, max_iter=1000)
        scores = cross_validate(r, X_transformed, y, cv = 7, scoring ='neg_root_mean_squared_error')
        error[i,j] = statistics.mean(-scores['test_score'])
        if error[i,j] <= min_error:
            best_alpha = alphas[j]
            best_l1_ratios = l1_ratios[i]

print(best_alpha)
print(best_l1_ratios)


plt.imshow(error, cmap='viridis')
plt.colorbar()
plt.show()

r_best = linear_model.ElasticNet(alpha=0.2, l1_ratio=0.8, fit_intercept=False, max_iter=1000)
r_best.fit(X_transformed, y)

np.savetxt('sub.csv', r_best.coef_, newline='\n')






