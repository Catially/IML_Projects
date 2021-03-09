import csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import statistics


with open('train.csv', 'r') as traindata:
    reader = csv.reader(traindata)
    header_row = next(reader)
    X = []
    y = []
    for row in reader:
        X.append(list(map(float, row[1:])))
        y.append(float(row[0]))

# Normalize X
std = StandardScaler()
X_tr = std.fit_transform(X)

alphas = [0.1, 1, 10, 100, 200]
avg_scores = []

for alpha in alphas:
    ridgemodel = Ridge(alpha = alpha, tol = 1e-10)
    scores = cross_validate(ridgemodel, X_tr, y, cv = 10, scoring ='neg_root_mean_squared_error')
    avg_scores.append(statistics.mean(-scores['test_score']))

with open('sub2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in range(len(avg_scores)):
        writer.writerow([avg_scores[row],])