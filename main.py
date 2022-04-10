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


def simple_plot_log_regression(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid()

    X = list(x)
    Y = list(y)
    plt.plot(X, Y, 'o')

    fit = np.polyfit(np.log(X), Y, deg=1)
    plt.plot(X, fit[0] * np.log(X) + fit[1])
    plt.show()


def simple_hist(x, xlabel, title):
    n, bins, patches = plt.hist(x, int(len(x) / 3), density=True, facecolor='g', alpha=0.75)
    avg = sum(x) / len(x)
    plt.axvline(x=avg, color='r', label='Average = {}'.format(avg))

    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


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

    fig, ax = plt.subplots()
    ax.set_xlabel('V14', fontsize=15)
    ax.set_ylabel('V17', fontsize=15)
    ax.set_title('Analysis of Features Correlated to Class')
    colors = ['r', 'g']
    targets = [0.0, 1.0]

    for target, color in zip(targets, colors):
        indicesToKeep = df['Class'] == target
        ax.scatter(df.loc[indicesToKeep, 'V14']
                   , df.loc[indicesToKeep, 'V17']
                   , c=color
                   , s=50)

    ax.legend(targets)
    ax.grid()

    plt.show()


def get_pca_tranformed(feature_values, features):
    # Transform data
    features_df = pd.DataFrame(data=feature_values, columns=features)
    time_amt_col = features_df.loc[:, ['Time', 'Amount']]
    pca_feat_col = features_df.loc[:, [x for x in features if x not in ['Time', 'Amount']]]
    pca_time_amount = StandardScaler().fit_transform(time_amt_col)

    return pd.concat([pd.DataFrame(data=pca_time_amount, columns=['Time', 'Amount']),
                           pca_feat_col], axis=1)


def show_pca(feature_values, features,  target_values):
    # Transform data
    X = get_pca_tranformed(feature_values, features).values[:, :]


    # Find principal components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])

    #print(principal_df.head(30))

    # Visualize Principal Components
    final_df = pd.concat([principal_df, pd.DataFrame(data=target_values, columns=['target'])], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0.0, 1.0]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'PC 1']
                   , final_df.loc[indicesToKeep, 'PC 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()

    plt.show()


    #3d scatterplot
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)

    ax.set_title('3 component PCA', fontsize=20)
    targets = [0.0, 1.0]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter3D(final_df.loc[indicesToKeep, 'PC 1'],
                   final_df.loc[indicesToKeep, 'PC 2'],
                   final_df.loc[indicesToKeep, 'PC 3'],
                   c=color,
                   s=50)
    ax.legend(targets)
    #ax.grid()

    plt.show()'''


    print(sum(pca.explained_variance_ratio_))


def find_best_num_estimators_random_forest(X, Y, cv):
    recalls = list()
    precisions = list()
    n_range = range(1, 26, 1)

    for n in n_range:
        print('n_range:', n)
        clf = RandomForestClassifier(n_estimators=n, max_depth=7, max_features=12)
        scores = cross_validate(clf, X, Y, cv=cv, scoring=['recall', 'precision'])
        r_scores = scores['test_recall']
        p_scores = scores['test_precision']
        recalls.append(sum(r_scores)/len(r_scores))
        precisions.append(sum(p_scores)/len(p_scores))

    simple_plot_log_regression(list(n_range), recalls, 'n_estimators', 'recall score', 'Performance of Random Forest, max_depth=5')
    simple_plot_log_regression(list(n_range), precisions, 'n_estimators', 'recall score', 'Performance of Random Forest, max_depth=5')


def find_best_max_depth_random_forest(X, Y, cv):
    recalls = list()
    precisions = list()
    f1s = list()
    n = 15
    n_range = range(1, n + 1, 1)

    for n in n_range:
        print('n_range:', n)
        clf = RandomForestClassifier(n_estimators=25, max_depth=n, random_state=42)
        scores = cross_validate(clf, X, Y, cv=cv, scoring=['recall', 'precision', 'f1'])
        r_scores = scores['test_recall']
        p_scores = scores['test_precision']
        f1_scores = scores['test_f1']
        recalls.append(sum(r_scores)/len(r_scores))
        precisions.append(sum(p_scores)/len(p_scores))
        f1s.append(sum(f1_scores)/len(f1_scores))

    simple_plot_log_regression(list(n_range), recalls, 'max_depth', 'recall score', 'Performance of Random Forest, 25 estimators')
    simple_plot_log_regression(list(n_range), precisions, 'max_depth', 'precision score', 'Performance of Random Forest, 25 estimators')
    simple_plot_log_regression(list(n_range), f1s, 'max_depth', 'f1 score', 'Performance of Random Forest, 25 estimators')


def find_best_max_features_random_forest(X, Y, cv):
    recalls = list()
    precisions = list()
    f1s = list()
    n = 15
    n_range = range(1, n + 1, 1)

    for n in n_range:
        print('n_range:', n)
        clf = RandomForestClassifier(n_estimators=25, max_depth=7, max_features=n, random_state=42)
        scores = cross_validate(clf, X, Y, cv=cv, scoring=['recall', 'precision', 'f1'])
        r_scores = scores['test_recall']
        p_scores = scores['test_precision']
        f1_scores = scores['test_f1']
        recalls.append(sum(r_scores)/len(r_scores))
        precisions.append(sum(p_scores)/len(p_scores))
        f1s.append(sum(f1_scores)/len(f1_scores))

    simple_plot_log_regression(list(n_range), recalls, 'max_depth', 'recall score', 'Performance of Random Forest, 25 estimators, max-depth = 7')
    simple_plot_log_regression(list(n_range), precisions, 'max_depth', 'precision score', 'Performance of Random Forest,  max-depth = 7')
    simple_plot_log_regression(list(n_range), f1s, 'max_depth', 'f1 score', 'Performance of Random Forest,  max-depth = 7')


def check_model_consistency_histo(x, y, splits, clf):
    ss = ShuffleSplit(n_splits=splits, test_size=0.2)

    scores = cross_validate(clf, x, y, cv=ss, scoring=['recall', 'precision', 'f1'])
    r_scores = scores['test_recall']
    p_scores = scores['test_precision']
    f1_scores = scores['test_f1']

    avg_recall = sum(r_scores) / len(r_scores)
    avg_precision = sum(p_scores) / len(p_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print('Avg Recall', avg_recall)
    print('Avg Precision', avg_precision)
    print('Avg F1', avg_f1)

    simple_hist(r_scores, 'Recall', 'Probability of Recall')
    simple_hist(p_scores, 'Precision', 'Probability of Precision')
    simple_hist(f1_scores, 'F1', 'Probability of F1')


def show_metrics(y_true, y_pred):
    print('accuracy', metrics.accuracy_score(y_true, y_pred))
    print('error rate', metrics.mean_absolute_percentage_error(y_true, y_pred))
    print('recall', metrics.recall_score(y_true, y_pred))
    print('precision positive', metrics.precision_score(y_true, y_pred))
    print('precision negative', metrics.precision_score(y_true, y_pred, pos_label=0))


datasource_loc = "creditcard.csv"
df = pd.read_csv(datasource_loc)
#show_correlation_heatmap(df)

df_cls_corr = df.loc[:, ['V14', 'V17']]
df_feat_corr = df.loc[:, [x for x in df.columns.tolist() if x not in ['Time', 'V2', 'V5', 'V7', 'V20', 'Class']]]
#df_feat_corr = df.loc[:, [x for x in df.columns.tolist() if x not in ['Time', 'Amount', 'Class']]]

#X = df_cls_corr.values[:, :]
X = df_feat_corr.values[:, :]
#X = df.values[:, 0:30]
Y = df.values[:, 30]

#target_df = df.loc[:, 'Class'].to_frame()
#pca_features_df = get_pca_tranformed(X, df.columns[:-1])
#pca_df = pd.concat([pca_features_df, target_df], axis=1)
#X = pca_df.values[:, 0:30]
#Y = pca_df.values[:, 30]

#show_correlation_heatmap(pca_df)


#show_pca(X, df.columns[:-1], Y)

#clf = RandomForestClassifier(n_estimators=25, max_depth=7, max_features=12, n_jobs=6)
# clf = LogisticRegression(max_iter=10000)
# clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=3, random_state=0)
#clf = SVC()



# clf = clf.fit(X, Y)
# print(clf.predict([[4462, -2.303349568, 1.75924746, -0.359744743, 2.330243051, -0.821628328, -0.075787571, 0.562319782,
#                    -0.698826069, -2.282193829, -4.781830856, -2.615664945, -1.334441067, -0.430021867, -0.294166318,
#                    -0.932391057, 0.172726296, -0.087329538, -0.156114265, -0.542627889, 0.039565989, -0.153028797,
#                    239.93]]))

# Y_pred = clf.predict(X)
# show_metrics(Y, Y_pred)
# tss = TimeSeriesSplit(n_splits=5)
ss = ShuffleSplit(n_splits=20, test_size=0.2)

#scores = cross_validate(clf, X, Y, cv=ss, scoring=['recall', 'precision'])
#find_best_num_estimators_random_forest(X, Y, ss)
#find_best_max_depth_random_forest(X, Y, ss)
#find_best_max_features_random_forest(X, Y, ss)

clf = RandomForestClassifier(n_estimators=25, max_depth=7, max_features=12, n_jobs=6)

check_model_consistency_histo(X, Y, 200, clf)

#print(scores)
#print('Avg Recall', sum(scores['test_recall']) / len(scores['test_recall']))
#print('Avg Precision', sum(scores['test_precision']) / len(scores['test_precision']))
