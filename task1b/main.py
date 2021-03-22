
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
import numpy as np
import csv
import statistics

# Read training data
with open('train.csv', 'r') as file:
    reader = csv.reader(file)
    header_row = next(reader)
    X_train = []
    y_train = []
    for row in reader:
        X_train.append(list(map(float, row[2:])))
        y_train.append(float(row[1]))

X_lin = np.array(X_train) 
X_quad = np.power(X_train, 2) 
X_exp = np.exp(X_train)
X_cos = np.cos(X_train)
X_const = np.array([1] * len(X_train))
X = np.column_stack((X_lin, X_quad, X_exp, X_cos, X_const))
y = np.array(y_train)

# model = linear_model.LinearRegression()
# model.fit(X, y)

# ridgemodel = Ridge(alpha = 0.001)
# ridgemodel.fit(X,y)

Lassomodel = Lasso(alpha = 0.001)
Lassomodel.fit(X,y)

print(Lassomodel.coef_)

'''
alphas = [0.001, 0.01, 0.1]
avg_scores = []

for alpha in alphas:
    lassomodel = Lasso(alpha = alpha)
    scores = cross_validate(lassomodel, X, y, cv = 7, scoring ='neg_root_mean_squared_error')
    avg_scores.append(statistics.mean(-scores['test_score']))

print(avg_scores)
'''


with open('sub.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(Lassomodel.coef_)):
        writer.writerow([Lassomodel.coef_[i],])
