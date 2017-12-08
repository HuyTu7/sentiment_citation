from __future__ import print_function, division
from collections import Counter
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Learners(object):
    def __init__(self, clf, train_X, train_Y, predict_X, predict_Y, goal):
        """
        :param clf: classifier, SVM, etc...
        :param train_X: training data, independent variables.
        :param train_Y: training labels, dependent variables.
        :param predict_X: testing data, independent variables.
        :param predict_Y: testing labels, dependent variables.
        :param goal: the objective of your tuning, F, precision,....
        """
        self.train_X = train_X.tolist()
        self.train_Y = train_Y
        self.predict_X = predict_X
        self.predict_Y = predict_Y
        self.goal = goal
        self.param_distribution = self.get_param()
        self.learner = clf
        self.confusion = None
        self.params = None
        self.scores = None

    def learn(self, F, **kwargs):
        """
        :param F: a dict, holds all scores, can be used during debugging
        :param kwargs: a dict, all the parameters need to set after tuning.
        :return: F, scores.
        """
        self.learner.set_params(**kwargs)
        clf = self.learner.fit(self.train_X, self.train_Y)
        predict_result = []
        predict_Y = []
        for predict_X, actual in zip(self.predict_X, self.predict_Y):
            try:
                _predict_result = clf.predict(predict_X.reshape(1, -1))
                predict_result.append(_predict_result[0])
                predict_Y.append(actual)
            except:
                print("one pass")
                continue
        self.scores = self._Abcd(predict_result, predict_Y, F)
        self.confusion = classification_report(predict_Y, predict_result, digits=3)
        self.params = kwargs
        return self.scores

    def _Abcd(self, predicted, actual, F):
        """
        :param predicted: predicted results(labels)
        :param actual: actual results(labels)
        :param F: previously got scores
        :return: updated scores.
        """
        def calculate(scores):
            for i, v in enumerate(scores):
                F[uni_actual[i]] = F.get(uni_actual[i], []) + [v]
            freq_actual = [count_actual[one] / len(actual) for one in uni_actual]
            F["mean"] = F.get("mean", []) + [np.mean(scores)]
            F["mean_weighted"] = (F.get("mean_weighted", []) +
                                    [np.sum(np.array(scores) * np.array(freq_actual))])
            return F

        def micro_cal(goal="Micro_F"):
            TP, FN, FP = 0, 0, 0
            for each in confusion_matrix_all_class:
                TP += each.TP
                FN += each.FN
                FP += each.FP
            PREC = TP / (TP + FP)
            PD = TP / (TP + FN)
            F[goal] = F.get(goal, []) + [round(2.0 * PREC * PD / (PD + PREC),3)]
            return F

        def macro_cal(goal="Macro_F"):
            PREC_sum, PD_sum = 0, 0
            for each in confusion_matrix_all_class:
                PD_sum += each.stats()[0]
                PREC_sum += each.stats()[2]
            PD_avg = PD_sum / len(uni_actual)
            PREC_avg = PREC_sum / len(uni_actual)
            F[goal] = F.get(goal, []) + \
                      [round(2.0 * PREC_avg * PD_avg / (PREC_avg + PD_avg), 3)]
            return F

        _goal = {"PD": 0, "PF": 1, "PREC": 2, "ACC": 3, "F": 4, "G": 5, "Macro_F": 6, "Micro_F": 7}

        abcd = ABCD(actual, predicted)
        uni_actual = list(set(actual))

        count_actual = Counter(actual)
        if "Micro" in self.goal or "Macro" in self.goal:
            confusion_matrix_all_class = [each for each in abcd()]
            if "Micro" in self.goal:
                return micro_cal()
            else:
                return macro_cal()
        else:
            score_each_klass = [k.stats()[_goal[self.goal]] for k in abcd()]
            return calculate(score_each_klass)

    def get_param(self):
        raise NotImplementedError("You should implement get_param function")


class SK_SVM(Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = SVC()
        super(SK_SVM, self).__init__(clf, train_x, train_y, predict_x, predict_y,
                                                                 goal)

    def get_param(self):
        tunelst = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "C": [1, 100], #np.logspace(-9, 9, num=10, base=10),
                    "coef0": [0.0, 1],
                    #"gamma": [0.0, 1],
                    "random_state": [42, 42]}
        return tunelst


class SK_RF(Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = RandomForestClassifier()
        super(SK_RF, self).__init__(clf, train_x, train_y, predict_x, predict_y,
                                                                 goal)

    def get_param(self):
        tunelst = {"max_features": [1, 100],
                    "max_leaf_nodes": [2, 51],
                    "min_samples_split": [2, 21],
                    "min_samples_leaf": [2, 21],
                    "n_estimators": [50, 151],
                    "random_state": [42, 42]}
        return tunelst


class SK_CART(Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = tree.DecisionTreeClassifier()
        super(SK_CART, self).__init__(clf, train_x, train_y, predict_x, predict_y,
                                                                 goal)

    def get_param(self):
        tunelst = {"max_features": [1, 100],
                    "max_depth": [1, 51],
                    "min_samples_split": [2, 21],
                    "min_samples_leaf": [1, 21],
                    "random_state": [79, 79]}
        return tunelst


class SK_KNN(Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = KNeighborsClassifier()
        super(SK_KNN, self).__init__(clf, train_x, train_y, predict_x, predict_y,
                                                                 goal)

    def get_param(self):
        tunelst = {"n_neighbors": [2, 10],
                   "weights": ["uniform", "distance"],
                   "random_state": [49, 49]}
        return tunelst


class counter():
    def __init__(self, before, after, indx):
        self.indx = indx
        self.actual = before
        self.predicted = after
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        for a, b in zip(self.actual, self.predicted):
            if a == indx and b == indx:
                self.TP += 1
            elif a == b and a != indx:
                self.TN += 1
            elif a != indx and b == indx:
                self.FP += 1
            elif a == indx and b != indx:
                self.FN += 1
            elif a != indx and b != indx:
                pass

    def stats(self):
        pd, pf, prec, F, G, acc = 0, 0, 0, 0, 0,0
        if self.TP + self.FN:
            pd = self.TP / (self.TP + self.FN)
        if self.FP+self.TN:
            pf = self.FP/(self.FP+self.TN)
        if self.TP+self.FP:
            prec = self.TP/(self.TP+self.FP)
        if self.TP+self.FP+self.TN+self.TP:
            acc = (self.TP +self.TN)/(self.TP+self.TN+self.FP+self.FN)
        if pd+prec:
            F = 2*pd*prec/(pd+prec)
        if pd+(1-pf):
            G = 2*pd*(1-pf)/(pd+1-pf)
        return pd, pf, prec, acc, F, G


class ABCD():
    "Statistics Stuff, confusion matrix, all that jazz..."

    def __init__(self, before, after):
        self.actual = before
        self.predicted = after

    def __call__(self):
        uniques = set(self.actual)
        for u in list(uniques):
            yield counter(self.actual, self.predicted, indx=u)

