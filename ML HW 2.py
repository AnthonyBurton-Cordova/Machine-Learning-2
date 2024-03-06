# The goal of this homework is to create a grid search first using just SKLearn and then later use the Optuna framework.


# Adapt this code below to run your analysis.
# Each part should require a single week

# Part 1
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression,
# each with 3 different sets of hyper parrameters for each
#


# Part 2
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data to work with
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Part 3
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function in sklearn

# Extra credit.
# Use Optuna to optimize a NN on a data set.

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


#Iris data
iris = datasets.load_iris()
M = iris.data
L = iris.target
n_folds = 5
data = (M, L, n_folds)

clfs = [RandomForestClassifier, GradientBoostingClassifier, LogisticRegression]

paramDic = {"RandomForestClassifier":{"n_estimators": [100, 200, 500], 
                                      "max_depth": [5,8,10]}, 
            "GradientBoostingClassifier": {"learning_rate": [.1, .01, .5],
                                           "n_estimators": [100, 200, 500]},
            "LogisticRegression":{"fit_intercept": [True, False],
                                  "penalty": ["l2", None]}}

for model in clfs:
   model_name = str(model).rsplit('.',1)[1][:-2]
   for arg_values in product(*[x for x in paramDic[model_name].values()]):
    arg_dic = {}
    for k, v in zip(paramDic[model_name].keys(), arg_values):
      arg_dic[k] = v
    print(arg_dic)

def run(a_clf, data, clf_hyper, n_folds=5):
    M, L = data  # Unpacking the data
    kf = KFold(n_splits=n_folds)
    results = []  # Storing results from each fold

    for train_index, test_index in kf.split(M):
        clf = a_clf(**clf_hyper)  # Creating the classifier with the given hyperparameters
        clf.fit(M[train_index], L[train_index])
        pred = clf.predict(M[test_index])
        # Calculating metrics
        acc = accuracy_score(L[test_index], pred)
        f1 = f1_score(L[test_index], pred, average='weighted')  # Using weighted for multi-class classification
        prec = precision_score(L[test_index], pred, average='weighted')  # Using weighted for multi-class classification
        results.append({"accuracy": acc, "f1": f1, "precision": prec})
    return np.mean([res["accuracy"] for res in results]), np.mean([res["f1"] for res in results]), np.mean([res["precision"] for res in results])

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
        
        # Example data placeholder, replace with your actual data
        data = (M, L)
        accuracies, f1, precision = run(clf, data, arg_dic)
        
        print(f"Classifier: {clf_name}, Parameters: {arg_dic}, Accuracy: {accuracies}, F1 Score: {f1}, Precision: {precision}")





results = run(RandomForestClassifier, data, clf_hyper={})

print(results)