import pandas as pd
import numpy as np
import math
import pickle

from random import randint, random, seed
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.neighbors import NearestNeighbors


class NUM:
    def __init__(self):
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.sd = 0
        self.hi = -math.exp(32)
        self.lo = math.exp(32)
        self.w = 1

    def updates(self, t, f):
        f = f or (lambda x: x)
        for _, one in enumerate(t):
            self.update(f(one))
        return self

    def norm(self, x):
        return (x - self.lo) / (self.hi - self.lo + math.exp(-32))

    def update(self, x):
        self.n = self.n + 1
        if x < self.lo:
            self.lo = x
        if x > self.hi:
            self.hi = x
        delta = x - self.mu
        self.mu += delta / self.n
        self.m2 += delta * (x - self.mu)
        if self.n > 1:
            self.sd = (self.m2 / (self.n - 1)) ** 0.5


def load_data(fpath, test, col_names=[]):
    """
    Loading Data

    :param fpath: string, path to data
    :param col_names: list of names, names of the columns for the dataset
    :param test: boolean, test or train data
    :return:
    """
    df = pd.read_csv(fpath, header=None, sep=',')
    df = df.dropna()
    if col_names:
        df.columns = col_names
    if not test:
        y = df.iloc[:, 0]
        X = df.drop(df.columns[[0]], axis=1)
        klass_dist(y)
        return X, y
    else:
        return df


def pickle_operating(fname, item, flag):
    # save or load the pickle file.
    file_name = '%s.pickle' % fname
    print(file_name)
    if flag == 1:
        with open(file_name, 'rb') as fs:
            item = pickle.load(fs)
            return item
    else:
        with open(file_name, 'wb') as fs:
            pickle.dump(item, fs, protocol=pickle.HIGHEST_PROTOCOL)


def klass_dist(labels):
    """
    Show distribution of lables across the dataset
    :param labels:
    """
    y = {}
    for l in labels:
        if l in y.keys():
            y[l] += 1
        else:
            y[l] = 0
    print("Class Distribution: ", y)
    print("\n")


def normalize_data(data):
    """
    Normalizing the Dataset

    :param data: the dataset to be normalized
    :return:
    """
    features_summary = {}
    normalized_df = pd.DataFrame(columns=data.columns)
    for column in data:
        normalized_col = []
        features_summary[column] = NUM()
        numeric_col = data[column].astype(float).values
        features_summary[column].updates(numeric_col, None)
        for entry in numeric_col:
            normalized_col.append(features_summary[column].norm(entry))
        normalized_df[column] = normalized_col
    return features_summary, normalized_df


def pca(X_train, n_feature):
    """
    Principal Component Analysis

    :param X_train: train part of the data
    :param n_feature: number of the features the user want to keep
    :return: the pca model after fitting to the train data
    """
    pca = PCA(n_components=n_feature)
    pca.fit(X_train)
    return pca


def rfe(learner, n_feature):
    """
    Recursive Feature Elimination

    :param learner: the machine learning model
    :param n_feature: number of the features the user want to keep
    :return: the rfe model after fitting to the learner
    """
    rfe = RFE(learner, n_feature)
    return rfe


class SMOTE():
    def execute(self, l, samples=[], labels=[]):
        np.random.seed(7)
        seed(7)
        return self.balance(samples, labels, r=int(l[0]), neighbors=int(l[1]))

    def smote(self, data, num, k=3, r=1):
        corpus = []
        if len(data) < k:
            k = len(data) - 1
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
        distances, indices = nbrs.kneighbors(data)
        for i in range(0, num):
            mid = randint(0, len(data) - 1)
            nn = indices[mid, randint(1, k - 1)]
            datamade = []
            for j in range(0, len(data[mid])):
                gap = random()
                datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
            corpus.append(datamade)
        corpus = np.array(corpus)
        corpus = np.vstack((corpus, np.array(data)))
        return corpus

    def balance(self, data_train, train_label, r=0, neighbors=0):
        """

        :param data_train:
        :param train_label:
        :param m:
        :param r:
        :param neighbors:
        :return:
        """
        pos_train = []
        neg_train = []
        obj_train = []
        for j, i in enumerate(train_label):
            if i == "o":
                obj_train.append(data_train[j])
            elif i == "p":
                pos_train.append(data_train[j])
            else:
                neg_train.append(data_train[j])
        pos_train = np.array(pos_train)
        neg_train = np.array(neg_train)
        obj_train = np.array(obj_train)
        data = [pos_train, obj_train, neg_train]
        mean_len = int(math.ceil((len(pos_train) + len(neg_train) + len(obj_train)) / 3))
        print("Mean Length: ", mean_len)
        for i in range(len(data)):
            if len(data[i]) < mean_len:
                m = mean_len - len(data[i])
                data[i] = self.smote(data[i], m, k=neighbors, r=r)
            else:
                m = mean_len
                data[i] = data[i][np.random.choice(len(data[i]), m, replace=False)]
        data_train_smoted = np.vstack((data[0], data[1], data[2]))
        label_train_smoted = ["p"] * len(data[0]) + ["o"] * len(data[1]) + ["n"] * len(data[2])
        return data_train_smoted, label_train_smoted


