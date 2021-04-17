import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import xgboost as xgb

# read data
df_train = pd.read_csv('./datasets/train_features.csv')
df_label = pd.read_csv('./datasets/train_labels.csv')
df_test = pd.read_csv('./datasets/test_features.csv')

# preprocess
def preprocess(df):
    pid = df['pid'].drop_duplicates().to_numpy()
    df = df.drop(columns=['Time'])
    df = df.groupby(['pid'], sort=False).mean()
    df = df.fillna(df.mean())
    transformer = StandardScaler()
    X = transformer.fit_transform(df)
    return pid, X

_, X_train = preprocess(df_train)
labels = list(df_label.columns)
labels.remove('pid')
pid_test, X_test = preprocess(df_test)
y_pred = {'pid': pid_test}
print('Preprocessed!')

# # Task 1 & 2
# for i in range(0, 10):
#     label = labels[i]
#     y_train = df_label[label]

#     # find imbalanced data
#     neg, pos = np.bincount(y_train)
#     total = neg + pos
#     print(label, ':\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

#     clf = MLPClassifier(max_iter = 500, hidden_layer_sizes = (10,10,5))

#     # count = Counter(y_train)
#     # imbalance = count[0] / count[1]
#     # clf = xgb.XGBClassifier(scale_pos_weight=imbalance, eval_metric = 'auc')

#     # selector = SelectKBest()
#     # X_train = selector.fit_transform(X_train, y_train)
#     # X_test = selector.transform(X_test)
#     # clf = svm.SVC(class_weight='balanced', probability=True)

#     cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
#     print('CV_score: ', np.mean(cv_score), ' \n')

#     # model = clf.fit(X_train, y_train)
#     # prob = model.predict_proba(X_test)
#     # prob = np.array(prob[:, 1])
#     # y_pred[label] = prob

#     print(label, ': finished!')


# Task 3
for i in range(11, len(labels)):
    label = labels[i]
    y_train = df_label[label]

    # model = xgb.XGBRegressor()
    model = MLPRegressor(max_iter = 500, hidden_layer_sizes = (10,10,5))

    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print('CV_score: ', np.mean(cv_score), ' \n')

    # model = clf.fit(X_train, y_train)
    # y_pred_label = model.predict(X_test)
    # y_pred[label] = y_pred_label

    print(label, ': finished!')

# df_pred = pd.DataFrame(y_pred)

# compression_options = dict(method='zip', archive_name='prediction.csv')
# df_pred.to_csv('prediction.zip', index=False, float_format='%.3f', compression=compression_options)
# print('All finished!')