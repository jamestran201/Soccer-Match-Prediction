import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV, chi2, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, accuracy_score
from itertools import cycle
import matplotlib.pyplot as plt

def plot_precision_recall_curve(n_classes, precision, recall):
    # Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange'])

    plt.figure(figsize=(7, 8))
    lines = []
    labels = []

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall')

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0}'.format(i))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()

def show_classification_metrics(classifier, features, y_true, n_classes):
    prob = None
    y_pred = classifier.predict(features)
    
    try:
        prob = classifier.predict_proba(features)
    except:
        pass
    
    if prob is None:
        try:
            prob = classifier.decision_function(features)
        except:
            print("Cannot get class probability or decision function from classifier")
            return
    

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], prob[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),prob.ravel())
    average_precision["micro"] = average_precision_score(y_true, prob,average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    print()
    
    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_weighted = precision_score(y_true, y_pred, average="weighted")
    precision_macro = precision_score(y_true, y_pred, average="macro")

    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_weighted = recall_score(y_true, y_pred, average="weighted")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("Accuracy score: {:.3f}".format(accuracy_score(y_true, y_pred)))
    print()
    
    print("Precision score (micro): {:.3f}".format(precision_micro))
    print("Recall score (micro): {:.3f}".format(recall_micro))
    print("F1 score (micro): {:.3f}".format(f1_micro))
    print()
    print("Precision score (weighted): {:.3f}".format(precision_weighted))
    print("Recall score (weighted): {:.3f}".format(recall_weighted))
    print("F1 score (weighted): {:.3f}".format(f1_weighted))
    print()
    print("Precision score (macro): {:.3f}".format(precision_macro))
    print("Recall score (macro): {:.3f}".format(recall_macro))
    print("F1 score (macro): {:.3f}".format(f1_macro))
    
    plot_precision_recall_curve(n_classes, precision, recall)

# Read input files
train_df = pd.read_csv("Data/formated.csv")
test_df = pd.read_csv("Data/formatedTesting.csv")

# Remove whitespace in column names
train_df.columns = [col.strip() for col in df.columns]
test_df.columns = [col.strip() for col in test_df.columns]

# Delete duplicate columns
del train_df["e1"]
del train_df["e1.1"]
del test_df["e1"]
del test_df["e1.1"]

## Split the dataframe into features and labels arrays
train_features_array = train_df.iloc[:, :-1].values
train_labels_array = train_df.loc[:, "target Feature"].values
train_labels_binarized = label_binarize(train_labels_array, classes=[0,1,2])

test_features_array = test_df.iloc[:, :-1].values
test_labels_array = test_df.loc[:, "target Feature"].values
test_labels_binarized = label_binarize(test_labels_array, classes=[0,1,2])

# Feature selection using chi2 test
print("##############################################")
print("Performing feature selection and tuning for SVM with RBF kernel")
select_k_best = SelectKBest(chi2, k="all")
select_k_best.fit(train_features_array, train_labels_array)

feature_scores = pd.DataFrame({"feature": train_df.columns[:-1], "chi2_score": select_k_best.scores_,
                              "p_value": select_k_best.pvalues_})

print("Number of features where p_value < 0.05: {}\n".format(feature_scores[feature_scores["p_value"] < 0.05].shape[0]))
print("Top 10 features according to chi squared score:")
print(feature_scores[feature_scores["p_value"] < 0.05].sort_values("chi2_score", ascending=False)[:10])

# Define scorer for grid search
accuracy_scorer = make_scorer(accuracy_score)

# Run grid search to determine the best number of features and values for hyperparameters
pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', None),
    ('normalize', MinMaxScaler((0,1))),
    ('classify', SVC(kernel="rbf"))
])

N_FEATURES_OPTIONS = [10, 20, 30, 40]
C_OPTIONS = [1, 5, 10, 50, 100]
GAMMA_OPTIONS = [0.01, 0.05, 0.001, 0.005]
param_grid = [
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS
    },
]
reducer_labels = ['KBest(chi2)']

grid = GridSearchCV(pipe, cv=5, n_jobs=2, param_grid=param_grid)
grid.fit(train_features_array, train_labels_array)

n_features = grid.best_params_["reduce_dim__k"]
best_C = grid.best_params_["classify__C"]
best_gamma = grid.best_params_["classify__gamma"]

print("Cross validation best_accuracy score: {:.3f}".format(grid.best_score_))
print("Best C: {}".format(best_C))
print("Best gamma: {}".format(best_gamma))
print("Best number of features: {}".format(n_features))
print("Features selected:")
print(feature_scores[feature_scores["p_value"] < 0.05].sort_values("chi2_score", ascending=False)[:n_features])

# Taken from: https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py
mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(len(C_OPTIONS), len(GAMMA_OPTIONS), len(N_FEATURES_OPTIONS))
mean_scores = mean_scores.max(axis=(0,1))

bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature selection for SVM RBF")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 1))

plt.show()

# Train the SVM on the entire training set using the parameters determined by grid search
svm_pipe = Pipeline([
    ('reduce_dim', SelectKBest(chi2, k=n_features)),
    ('normalize', MinMaxScaler((0,1))),
    ('classify', SVC(kernel="rbf", C=best_C, gamma=best_gamma))
])

svm_pipe.fit(train_features_array, train_labels_array)

print("Accuracy score on test set: {:.3f}".format(svm_pipe.score(test_features_array, test_labels_array)))
print("##############################################")
print()

# Feature selection and tuning for SVM with linear kernel
print("Performing feature selection and tuning for SVM with Linear kernel")

# Taken from: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
normalizer = MinMaxScaler((0, 1))
normalizer.fit(train_features_array)

rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
rfecv.fit(normalizer.transform(train_features_array), train_labels_array)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

rfe_ranking = pd.DataFrame({"features": train_df.columns[:-1], "rfecv_rank": rfecv.ranking_})
print("Top 16 features according to RFECV")
print(rfe_ranking.sort_values("rfecv_rank")[:16])

selected_features = rfecv.transform(normalizer.transform(train_features_array))


# Find the best value of C for SVM with Linear kernel
param_grid = {
        "C": [1, 2, 3, 4, 5, 10, 50, 100]
    }

grid = GridSearchCV(SVC(kernel="linear"), cv=5, n_jobs=2, param_grid=param_grid)
grid.fit(selected_features, train_labels_array)

best_C = grid.best_params_["C"]

print("Cross validation best_accuracy score: {:.3f}".format(grid.best_score_))
print("Best C: {}".format(best_C))

# Train the SVM on the entire training set using the parameters determined by grid search
linear_svm = SVC(kernel="linear", C=best_C)
linear_svm.fit(selected_features, train_labels_array)

selected_test_features = rfecv.transform(normalizer.transform(test_features_array))

print("Accuracy score on test set: {:.3f}".format(linear_svm.score(selected_test_features, test_labels_array)))

feature_coefs_class_0 = rfe_ranking.sort_values("rfecv_rank")[:rfecv.n_features_]
feature_coefs_class_0 = feature_coefs_class_0.sort_index()
feature_coefs_class_0["coef"] = linear_svm.coef_[0]
feature_coefs_class_0["abs_coef"] = feature_coefs_class_0["coef"].abs()

print("Coefficients assigned to features sorted by absolute values")
print(feature_coefs_class_0.sort_values("abs_coef", ascending=False))
print("##############################################")

