import pandas as pd
from readarff import ReadData
import numpy as np
import scipy.sparse as ssparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as timer

readData = ReadData()
df = readData.csv_operating("./test_data", None)
labels = df.loc[:, "class"]
df = df.dropna()


def tokenizer(data, n_gram):
    tokenized_data = []

    if n_gram == 0:
        for r in data:
            tokenized_data.append(r.split())
    else:
        for r in data:
            str = [' '.join(r[i:i+n_gram]) for i in range(len(r)-n_gram)]
            tokenized_data.append(str)
    return tokenized_data


def processing_text(data, n_gram):
    count_vect = CountVectorizer(lowercase=True, ngram_range=(1, n_gram))
    tf_transformer = TfidfTransformer(use_idf=True)
    x_train_count = count_vect.fit_transform(data)
    x_train_tfidf = tf_transformer.fit_transform(x_train_count)
    return x_train_tfidf


def processing(df):
    trained_td = processing_text(df.loc[:, 'text'], 3)
    trained_tt = processing_text(df.loc[:, 'dependencies'], 1)
    readData.pickle_operating("trained_tdepend", trained_td, 0)
    readData.pickle_operating("trained_ttext", trained_tt, 0)


def load_trained():
    trained_td = readData.pickle_operating("trained_tdepend", None, 1)
    trained_tt = readData.pickle_operating("trained_ttext", None, 1)
    return trained_td, trained_tt


def experiment(trained_data, labels, clf):
    print trained_data.shape
    start = timer()
    scores = cross_validate(clf, trained_data, labels, scoring=['f1_micro', 'precision_micro', 'recall_micro', 'f1_macro'],
                            cv=5, return_train_score=False)
    duration = timer() - start
    print(duration)
    print(scores)


def read_CSV_sparseM():
    with open("sentences.csv") as f:
        ncols = len(f.readline().split(','))

    np_array = np.loadtxt("sentences.csv", delimiter=',', usecols=range(1, ncols))
    trained_data = ssparse.csr_matrix(np_array)
    return trained_data


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


def fit_save_model(clf, trained_data, labels, name):
    split = 0.3
    size = int(len(labels) * split)
    X_train, X_test, y_train, y_test = train_test_split(trained_data, labels, test_size=size)
    clf.fit(X_train, y_train)
    pickle_operating(name, clf, 0)


def learner(model):
    if model == 'svm':
        clf = SVC(C=1000, random_state=49)
    elif model == 'rf':
        clf = RandomForestClassifier(random_state=45)
    elif model == 'knn':
        clf = KNeighborsClassifier()
    elif model == 'cart':
        clf = tree.DecisionTreeClassifier(random_state=42)
    else:
        raise NameError('Unknown machine learning model. Please us one of: rf, svm, nb')
    return clf


if __name__ == "__main__":
    pca = False
    if(pca):
        df = pd.read_csv('sentences.csv', sep=',', header=None)
        labels = df.iloc[:, 0]
        trained_data = df.drop(df.columns[[0]], axis=1)
    else:
        trained_data = pickle_operating('X_pca_1000', None, 1)
        labels = pickle_operating('y_train', None, 1)
    model = "knn"
    clf = learner(model)
    fit_save_model(clf, trained_data, labels, model)
    experiment(trained_data, labels, clf)
