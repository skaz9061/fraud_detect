import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def show_correlation_heatmap(df):
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
#show_correlation_heatmap(df)

X = df.values[:, 0:30]
Y = df.values[:, 30]
clf = RandomForestClassifier(n_estimators=20)
clf = clf.fit(X, Y)
#print(clf.predict([[4462, -2.303349568, 1.75924746, -0.359744743, 2.330243051, -0.821628328, -0.075787571, 0.562319782,
#                    -0.698826069, -2.282193829, -4.781830856, -2.615664945, -1.334441067, -0.430021867, -0.294166318,
#                    -0.932391057, 0.172726296, -0.087329538, -0.156114265, -0.542627889, 0.039565989, -0.153028797,
#                    239.93]]))

Y_pred = clf.predict(X)
show_metrics(Y, Y_pred)




