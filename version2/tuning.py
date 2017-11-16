from __future__ import division, print_function
import random
import time
from features_engineer import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from learners import SK_SVM
from de_tuner import DE_Tune_ML


def tune_learner(learner, train_X, train_Y, tune_X,
                 tune_Y, goal, target_class=None):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :param target_class:
    :return:
    """
    if not target_class:
        target_class = goal
    clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
    tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
    tuner.Tune()
    return tuner.bestconf, tuner.bestscore, tuner.evaluation


def print_results(clfs):
    file_name = time.strftime("./%Y%m%d_%H:%M:%S.txt")
    content = ""
    for each in clfs:
        content += each.confusion
    with open(file_name, "w") as f:
        f.write(content)


def run_tuning_SVM(train_data, train_labels, learn_goal, repeats=1, fold=10, tuning=True):
    """
    :param train_data: np_array, training data
    :param train_labels: np_array, training labels
    :param learn_goal: string, tuning/learning goal(PD, PF, ACC, etc)
    :param repeats: int, number of repeats
    :param fold: int,number of folds
    :param tuning: boolean, tuning or not.
    :return: tuned_params: list of tuples(0: params, 1: score)
    """
    random.seed(7)
    split = 0.25
    size = int(len(train_labels) * split)
    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=size, random_state=5)
    balance_klass = SMOTE()
    learner = [SK_SVM][0]
    eval_measurements = ["PD", "PF", "PREC", "ACC", "F", "G", "Macro_F", "Micro_F"]
    if learn_goal in eval_measurements:
        goal = learn_goal
    else:
        goal = "Macro_F"
    F = {}
    clfs = []
    tuned_params = []
    l = {0: 0, 1: 2, 2: 5}
    for i in xrange(repeats):    # repeat n times here
        kf = StratifiedKFold(n_splits=fold, shuffle=True)
        for train_index, tune_index in kf.split(X_train, y_train):
            train_X = X_train[train_index]
            train_Y = y_train[train_index]
            l[0] = klass_bal_stats(train_Y)
            train_X, train_Y = balance_klass.execute(l, samples=train_X, labels=train_Y)
            tune_X = X_train[tune_index]
            tune_Y = y_train[tune_index]
            test_X = X_valid
            test_Y = y_valid
            params, score, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                            tune_Y, goal) if tuning else ({}, 0)
            clf = learner(train_X, train_Y, test_X, test_Y, goal)
            F = clf.learn(F, **params)
            clfs.append(clf)
            tuned_params.append((params, score))
    return sort_config(tuned_params, learn_goal)


def sort_config(tuned_params, goal):
    if goal == "PF":  # the less, the better.
        sort_params = sorted(tuned_params,
                         key=lambda x: x[1][goal])
    else:
        sort_params = sorted(tuned_params,
                         key=lambda x: x[1][goal], reverse=True)
    return sort_params


def klass_bal_stats(y_train):
    count = [0, 0, 0]
    for i in y_train:
        if i == "Obama":
            count[0] += 1
        elif i == "Romney":
            count[1] += 1
    count[2] = count[0] + count[1]
    mean = int(count[2]/2)
    return mean - count[0]


if __name__ == "__main__":
    tuned_params = run_tuning_SVM()
