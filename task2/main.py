import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn import linear_model


def features_engineering(df):
    features = [np.nanmax, np.nanmin, np.nanmean, np.nanmedian, np.nanstd]
    df = df.drop(columns=['Time'])
    data = df.groupby(['pid'], sort=False).agg(features) # feature augmentation
    data = data.fillna(data.median()) # fill nan with median
    # transformer = StandardScaler()
    transformer = RobustScaler()
    X = transformer.fit_transform(data)
    return X

# def features_engineering(data):
#     data = data.to_numpy()
#     n = 12
#     x = []
#     features = [np.nanmedian, np.nanmean, np.nanvar, np.nanmin,np.nanmax]

#     for index in range(int(data.shape[0] / n)):
#         assert data[n * index, 0] == data[n * (index + 1) - 1, 0], \
#         'Ids are {}, {}'.format(data[n * index, 0], data[n * (index + 1) - 1, 0])
#         patient_data = data[n * index: n * (index + 1), 2:]
#         feature_values = np.empty((len(features), data[:, 2:].shape[1]))
#         for i, feature in enumerate(features):
#             feature_values[i] = feature(patient_data, axis=0)
#         x.append(feature_values.ravel())

#     x = np.array(x)
#     imp = SimpleImputer(strategy='median')
#     x = imp.fit_transform(x)
#     tf = StandardScaler()
#     x = tf.fit_transform(x)
#     return x

# read data
df_train = pd.read_csv('./datasets/train_features.csv')
df_label = pd.read_csv('./datasets/train_labels.csv')
X_train = features_engineering(df_train)
labels = list(df_label.columns)
labels.remove('pid')

df_test = pd.read_csv('./datasets/test_features.csv')
X_test = features_engineering(df_test)
pid_test = df_test['pid'].drop_duplicates().to_numpy()
y_pred = {'pid': pid_test}
print('Preprocessed!\n')

# print(X_train)
# print(X_test)

# l2_regularization, learning_rate, max_depth
params = np.zeros((15, 3))
params[0] = [0.001, 0.05, 7]
params[1] = [0.1, 0.1, 4]
params[2] = [0.001, 0.05, 8]
params[3] = [0.01, 0.05, 9]
params[4] = [0.0001, 0.05, 4]
params[5] = [0.0, 0.1, 8]
params[6] = [0.001, 0.05, 3]
params[7] = [0.01, 0.05, 7]
params[8] = [0.01, 0.05, 4]
params[9] = [0.0, 0.05, 4]
params[10] = [0.0, 0.05, 3]
params[11] = [0.01, 0.05, 5]
params[12] = [0.0, 0.05, 9]
params[13] = [0.01, 0.15, 6]
params[14] = [0.0001, 0.05, 3]


# Task 1 & 2
for i in range(11):
    label = labels[i]
    y_train = df_label[label]

    # # grid search
    # parameters = {
    #     'learning_rate':[0.05, 0.10, 0.15, 0.20],
    #     'max_depth':[3, 4, 5, 6, 7, 8, 9], 
    #     'l2_regularization':[0.0, 0.0001, 0.001, 0.01, 0.1]
    # }

    # model = HistGradientBoostingClassifier(max_iter=300)

    # GDSCV = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='roc_auc', n_jobs=-1)
    # GDSCV.fit(X_train, y_train)
    # print(GDSCV.best_params_)

    # cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    # print("Cross-validation score is {score:.3f},"
    #     " standard deviation is {err:.3f}"
    #     .format(score = cv_score.mean(), err = cv_score.std()))

    model = HistGradientBoostingClassifier(max_iter=300, l2_regularization=params[i][0], learning_rate=params[i][1], max_depth=int(params[i][2]))

    model = model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    prob = np.array(prob[:, 1])
    y_pred[label] = prob

    print(label, ': finished!\n')


# Task 3
for i in range(11, len(labels)):
    label = labels[i]
    y_train = df_label[label]

    # # grid search
    # parameters = {
    #     'learning_rate':[0.05, 0.10, 0.15, 0.20],
    #     'max_depth':[3, 4, 5, 6, 7, 8, 9], 
    #     'l2_regularization':[0.0, 0.0001, 0.001, 0.01, 0.1]
    # }

    # model = HistGradientBoostingClassifier(max_iter=300)

    # GDSCV = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='r2', n_jobs=-1)
    # GDSCV.fit(X_train, y_train)
    # print(GDSCV.best_params_)

    # cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    # print("Cross-validation score is {score:.3f},"
    #     " standard deviation is {err:.3f}"
    #     .format(score = cv_score.mean(), err = cv_score.std()))

    model = HistGradientBoostingRegressor(max_iter=300, l2_regularization=params[i][0], learning_rate=params[i][1], max_depth=int(params[i][2]))

    model = model.fit(X_train, y_train)
    y_pred_label = model.predict(X_test)
    y_pred[label] = y_pred_label

    print(label, ': finished! \n')

df_pred = pd.DataFrame(y_pred)

compression_options = dict(method='zip', archive_name='prediction.csv')
df_pred.to_csv('prediction.zip', index=False, float_format='%.3f', compression=compression_options)

print('All finished!')