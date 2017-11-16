import pandas as pd
import numpy as np
import math

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


def load_data(fpath, col_names, test):
    """
    Loading Data

    :param fpath: string, path to data
    :param col_names: list of names, names of the columns for the dataset
    :param test: boolean, test or train data
    :return:
    """
    df = pd.read_csv(fpath, header=None, skiprows=1, index_col=False)
    df = df.dropna()
    df.columns = col_names
    if not test:
        X = df.drop('winner', axis=1)
        y = ['Obama' if w == "Barack Obama" else 'Romney' for w in df['winner'].values]
        klass_dist(y)
        return X, y
    else:
        return df


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
        return self.balance(samples, labels, m=int(l[0]), r=int(l[1]), neighbors=int(l[2]))

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

    def balance(self, data_train, train_label, m=0, r=0, neighbors=0):
        pos_train = []
        neg_train = []
        for j, i in enumerate(train_label):
            if i == "Obama":
                pos_train.append(data_train[j])
            else:
                neg_train.append(data_train[j])
        pos_train = np.array(pos_train)
        neg_train = np.array(neg_train)

        if len(pos_train) < len(neg_train):
            pos_train = self.smote(pos_train, m, k=neighbors, r=r)
            if len(neg_train) < m:
                m = len(neg_train)
            neg_train = neg_train[np.random.choice(len(neg_train), m, replace=False)]
        data_train1 = np.vstack((pos_train, neg_train))
        label_train = ["Obama"] * len(pos_train) + ["Romney"] * len(neg_train)
        return data_train1, label_train


