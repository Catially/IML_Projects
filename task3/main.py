import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

#------------------------------------------------------------------------
def features_engineering(data):
    X = []
    for i in range(len(data)):
        mutation = [char for char in data[i]]
        mutation_data = []
        for j in range(len(mutation)):
            AA_data = AA_detect(mutation[j])
            mutation_data = mutation_data + AA_data.tolist()
        X.append(mutation_data)
    X = np.array(X)
    return X

# Sequence
# R, H, K, D, E
# S, T, N, Q
# C, U, G, P
# A, I, L, M, F, W, Y, V
Sequence = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 
            'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

def AA_detect(sdata):
    booldata = np.zeros((21,), dtype=int)
    booldata[Sequence.index(sdata)] = 1
    return booldata

df_train = pd.read_csv('./datasets/train.csv')
data_train = np.array(df_train["Sequence"])
y_train = np.array(df_train["Active"])
df_test = pd.read_csv('./datasets/test.csv')
data_test = np.array(df_test["Sequence"])

X_train = features_engineering(data_train)
X_test = features_engineering(data_test)

print("Preprocessed!")

#------------------------------------------------------------------------

clf = HistGradientBoostingClassifier(learning_rate=0.15, 
                                              max_iter=200, 
                                              max_leaf_nodes=150, 
                                              min_samples_leaf=50,
                                              l2_regularization=1.0,
                                              scoring='f1')
model = clf
print("Running...")
cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
print("Cross-validation score is {score:.3f},"
    " standard deviation is {err:.3f}\n"
    .format(score = cv_score.mean(), err = cv_score.std()))

# -----------------------------------------------------------

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv('sub.csv', index=False, header=False)

print("All finished!")