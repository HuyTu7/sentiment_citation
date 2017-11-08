import pandas
from readarff import ReadData
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics

readData = ReadData()
df = readData.csv_operating("./test_data", None)
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
    # tokenized_text = tokenizer(df.loc[:, 'text'], 3)
    # tokenized_depencies = tokenizer(df.loc[:, 'dependencies'], 0)
    trained_td = processing_text(df.loc[:, 'text'], 3)
    trained_tt = processing_text(df.loc[:, 'dependencies'], 1)
    readData.pickle_operating("trained_tdepend", trained_td, 0)
    readData.pickle_operating("trained_ttext", trained_tt, 0)

def load_trained():
    trained_td = readData.pickle_operating("trained_tdepend", None, 1)
    trained_tt = readData.pickle_operating("trained_ttext", None, 1)
    return trained_td, trained_tt

def experiment(trained_data, labels):
    # random.seed(1)
    split = 0.3
    print trained_data.shape
    clf = SVC(C=1000, random_state=1, kernel="linear")
    scores = cross_val_score(clf, trained_data, labels, cv=3)
    print(scores)
    # random.shuffle(keys)
    size = int(len(labels) * split)
    X_train, X_test, y_train, y_test = train_test_split(trained_data, labels, test_size=size, random_state=1)
    clf.fit(X_train, y_train)
    print()
    print()
    print("Scores on test set: %s" % classification_report(y_test, clf.predict(X_test)))
    print()

processing(df)
trained_td, trained_tt = load_trained()
import scipy.sparse as ssparse
trained_data = ssparse.hstack([trained_tt, trained_td])
labels = df.loc[:, "class"]
experiment(trained_tt, labels)
experiment(trained_td, labels)
experiment(trained_data, labels)