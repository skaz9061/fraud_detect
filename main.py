import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, TimeSeriesSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def show_correlation_heatmap(df):
    corrs_abs = np.copy(abs(df.corr()))
    corrs_shp = corrs_abs.shape
    for r in range(corrs_shp[0]):
        for c in range(corrs_shp[1]):
            if c >= r:
                corrs_abs[r, c] = 0

    corrs_abs = np.where(corrs_abs < 0.3, 0, corrs_abs)
    corrs_cords = np.nonzero(corrs_abs)

    for tup in zip(corrs_cords[0].tolist(), corrs_cords[1].tolist()):
        print(tup, df.corr().iat[tup], df.columns[list(tup)])

    plt.imshow(df.corr())
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns, rotation='vertical')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.show()

def show_metrics(y_true, y_pred):
    print('accuracy', metrics.accuracy_score(y_true, y_pred))
    print('error rate', metrics.mean_absolute_percentage_error(y_true, y_pred))
    print('recall', metrics.recall_score(y_true, y_pred))
    print('precision positive', metrics.precision_score(y_true, y_pred))
    print('precision negative', metrics.precision_score(y_true, y_pred, pos_label=0))


datasource_loc = "creditcard.csv"
df = pd.read_csv(datasource_loc)
show_correlation_heatmap(df)

X = df.values[:, 0:30]
Y = df.values[:, 30]
# clf = RandomForestClassifier(n_estimators=20)
# clf = RandomForestClassifier(n_estimators=20, max_depth=3)
# clf = RandomForestClassifier(n_estimators=20, max_depth=5)
# clf = RandomForestClassifier(n_estimators=20, max_features=5, max_depth=5)
# clf = RandomForestClassifier(n_estimators=20, max_features=3, max_depth=5)
# clf = RandomForestClassifier(n_estimators=40, max_features=3, max_depth=5)
# clf = RandomForestClassifier(n_estimators=100, max_features=3, max_depth=5)
# clf = RandomForestClassifier(n_estimators=50, max_features=4, max_depth=5)
# clf = RandomForestClassifier(n_estimators=20, max_features=5, max_depth=5, random_state=0)
# clf = RandomForestClassifier(n_estimators=20, max_features='sqrt', max_depth=5, random_state=0)
# clf = RandomForestClassifier(n_estimators=20, max_features='log2', max_depth=5, random_state=0)
clf = RandomForestClassifier(n_estimators=20, max_features='auto', max_depth=5, random_state=0)
# clf = LogisticRegression()
# clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=3, random_state=0)
# clf = GradientBoostingClassifier(n_estimators=20)
# clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, max_depth=3, random_state=0)
# clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features='sqrt', max_depth=5, random_state=0)
# clf = clf.fit(X, Y)
# print(clf.predict([[4462, -2.303349568, 1.75924746, -0.359744743, 2.330243051, -0.821628328, -0.075787571, 0.562319782,
#                    -0.698826069, -2.282193829, -4.781830856, -2.615664945, -1.334441067, -0.430021867, -0.294166318,
#                    -0.932391057, 0.172726296, -0.087329538, -0.156114265, -0.542627889, 0.039565989, -0.153028797,
#                    239.93]]))

# Y_pred = clf.predict(X)
# show_metrics(Y, Y_pred)
# tss = TimeSeriesSplit(n_splits=5)
ss = ShuffleSplit(n_splits=5, test_size=0.2)

# scores = cross_validate(clf, X, Y, cv=ss, scoring=['accuracy', 'recall', 'precision'])

# print(scores)
