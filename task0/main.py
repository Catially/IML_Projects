import csv
import statistics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

with open('data/train.csv', 'r') as traindata:
    trainreader = csv.reader(traindata)
    trainheader = next(trainreader)
    trainx = []
    trainy = []
    for row in trainreader:
        trainrowdata = list(map(float, row[2:]))
        trainx.append(trainrowdata)
        trainy.append(float(row[1]))


with open('data/test.csv', 'r') as testdata:
    testreader = csv.reader(testdata)
    testheader = next(testreader)
    testx = []
    testy = []
    Id = []
    for row in testreader:
        Id.append(row[0])
        testrowdata = list(map(float, row[1:]))
        testx.append(testrowdata)
        testy.append(statistics.mean(testrowdata))

predy = testy

"""    
model = linear_model.LinearRegression()
model.fit(trainx, trainy)
predy = model.predict(testx)
"""

rmse = mean_squared_error(testy, predy)**0.5
print(rmse)

with open('pred.csv', 'w', newline = '') as pred:
    writer = csv.writer(pred)
    writer.writerow(['Id', 'y'])
    for row in range(len((predy))):
        writer.writerow([Id[row], predy[row]])