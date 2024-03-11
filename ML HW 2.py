# The goal of this homework is to create a grid search first using just SKLearn and then later use the Optuna framework.
# Adapt this code below to run your analysis.
# Each part should require a single week


#### Part 1 and 2

# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression,
# each with 3 different sets of hyper parrameters for each
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data to work with
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Import libraries
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from itertools import product
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split



# Import the Iris dataset
iris = datasets.load_iris()
M = iris.data
L = iris.target
n_folds = 5
data = (M, L, n_folds)

# Define the Classifiers I'll be using
clfs = [RandomForestClassifier, GradientBoostingClassifier, LogisticRegression]

# Creating a dictionary to store the classifiers and their respective hyperparameters 
paramDic = {"RandomForestClassifier":{"n_estimators": [100, 200, 500], 
                                      "max_depth": [5,8,10]}, 
            "GradientBoostingClassifier": {"learning_rate": [.1, .01, .5],
                                           "n_estimators": [100, 200, 500]},
            "LogisticRegression":{"fit_intercept": [True, False],
                                  "penalty": ["l2", None]}}

# Creating a loop to store each classifier and use each hyperparameter 
for model in clfs:
   model_name = str(model).rsplit('.',1)[1][:-2]
   for arg_values in product(*[x for x in paramDic[model_name].values()]):
    arg_dic = {}
    for k, v in zip(paramDic[model_name].keys(), arg_values):
      arg_dic[k] = v
    print(arg_dic)

# def run function
def run(a_clf, data, clf_hyper, n_folds=5):
    M, L = data
    kf = KFold(n_splits=n_folds)
    results = []
    for train_index, test_index in kf.split(M):
        clf = a_clf(**clf_hyper)
        clf.fit(M[train_index], L[train_index])
        pred = clf.predict(M[test_index])
        
        # I want to use accuracy, f1, and precision to score each classifier
        acc = accuracy_score(L[test_index], pred)
        f1 = f1_score(L[test_index], pred, average='weighted')  # Using weighted for multi-class classification
        prec = precision_score(L[test_index], pred, average='weighted')  # Using weighted for multi-class classification
        results.append({"accuracy": acc, "f1": f1, "precision": prec})
    return np.mean([res["accuracy"] for res in results]), np.mean([res["f1"] for res in results]), np.mean([res["precision"] for res in results])

# Now we need to train each classifier
def train_classifiers(clfs, paramDic, data, n_folds=5):
    results = {}
    for clf in clfs:
        clf_name = clf.__name__
        for arg_values in product(*paramDic[clf_name].values()):
            arg_dic = dict(zip(paramDic[clf_name].keys(), arg_values))
            avg_accuracy, avg_f1, avg_precision = run(clf, data, arg_dic, n_folds)
            results[(clf_name, tuple(arg_dic.items()))] = {"avg_accuracy": avg_accuracy, "avg_f1": avg_f1, "avg_precision": avg_precision}
    return results


# Loop through each classifier and their hyperparameter combinations
for clf in clfs:
    clf_name = clf.__name__
    for arg_values in product(*paramDic[clf_name].values()):
        arg_dic = dict(zip(paramDic[clf_name].keys(), arg_values))
        data = (M, L)
        accuracies, f1, precision = run(clf, data, arg_dic)
        
        print(f"Classifier: {clf_name}, Parameters: {arg_dic}, Accuracy: {accuracies}, F1 Score: {f1}, Precision: {precision}")

# Store the results from each classifier
results = train_classifiers(clfs, paramDic, data, n_folds)

# Plot the scores
def plot_metric(results, metric_name):
    labels = []
    scores = []
    for (clf_name, params), metrics in results.items():
        param_summary = ', '.join([f"{k.split('_')[-1]}:{v}" for k, v in params])
        labels.append(f"{clf_name}\n{param_summary}")
        if metric_name == 'accuracy':
            scores.append(metrics["avg_accuracy"])
        elif metric_name == 'f1':
            scores.append(metrics["avg_f1"])
        elif metric_name == 'precision':
            scores.append(metrics["avg_precision"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    rects = ax.bar(x, scores, label=metric_name.capitalize())

    # Making the charts readable by adding labels
    ax.set_ylabel(f'{metric_name.capitalize()} Scores')
    ax.set_title(f'{metric_name.capitalize()} by classifier and parameter combination')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects)
    fig.tight_layout()

# Plot the scores from each classifier
def plot_all_metrics_separately(results):
    plot_metric(results, 'accuracy')
    plt.show()
    plot_metric(results, 'f1')
    plt.show()
    plot_metric(results, 'precision')
    plt.show()

plot_all_metrics_separately(results)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(M, L, test_size=0.3, random_state=42)

#### Part 3

# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function in sklearn

# Parameter grid specifically for RandomForestClassifier
param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 8, 10, None]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV for RandomForestClassifier
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

# Perform the grid search on the training data
grid_search_rf.fit(X_train, y_train)

# Output the best parameters and the corresponding score
print("Best parameters found for RandomForestClassifier: ", grid_search_rf.best_params_)
print("Best cross-validation score: {:.3f}".format(grid_search_rf.best_score_))
