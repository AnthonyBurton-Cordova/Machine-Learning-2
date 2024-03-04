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
from sklearn.metrics import accuracy_score # Please select some other metircs in addtion to this.
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from itertools import product


#Example data
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
n_folds = 5

data = (M, L, n_folds)

clfs = [RandomForestClassifier, GradientBoostingClassifier, LogisticRegression]
paramDic = {"RandomForestClassifier":{"n_estimators": [100, 200, 500], 
                                      "max_depth": [5,8,10]}, 
            "GradientBoostingClassifier": {"learning_rate": [.1, .01, .5],
                                           "n_estimators": [100, 200, 500]},
            "LogisticRegression":{"fit_intercept": [True, False],
                                  "positive": [True, False]}}

for model in clfs:
   model_name = str(model).rsplit('.',1)[1][:-2]
   for arg_values in product(*[x for x in paramDic[model_name].values()]):
    arg_dic = {}
    for k, v in zip(paramDic[model_name].keys(), arg_values):
      arg_dic[k] = v
    print(arg_dic)


def train_classifiers(clf, data);
  results = {}
  for clf_name, hyperparams in paramDic.items():
    #paramDicK = paramDic.keys()
    #hyperparamsV = paramDic.values()
    for clf_name, hyperparams in hpnames.


hyperparamsV = paramDic.values()
hyperparamsV



def run(a_clf, data, clf_hyper, M, L, n_folds):
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret



results = run(RandomForestClassifier, data, clf_hyper={})

print(results)