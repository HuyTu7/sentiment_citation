import numpy as np
import pickle

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from features_engineer import load_data, normalize_data, pca
from tuning import run_tuning_SVM


def learner(model, tune):
    if model == 'svm':
        if tune:
            #winner_class = svm.SVC(kernel='sigmoid', C=36, coef0=0.08, gamma=0.23, random_state=1)
            #winner_class = svm.SVC(kernel='poly', C=11, coef0=0.307, gamma=0.43, random_state=1)
            winner_class = svm.SVC(kernel='rbf', C=38, coef0=0.715, gamma=0.354, random_state=1)
        else:
            winner_class = svm.SVC()
    elif model == 'rf':
        winner_class = RandomForestClassifier(n_estimators=10, random_state=1)
    elif model == 'nb':
        winner_class = GaussianNB()
    else:
        raise NameError('Unknown machine learning model. Please us one of: rf, svm, nb')
    return winner_class


def experiment(data, classes, model):
    scores = cross_val_score(model, data, classes, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    split = 0.3
    size = int(len(classes) * split)
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=size, random_state=1)
    model.fit(X_train, y_train)
    print()
    print()
    print("Scores on test set: %s" % classification_report(y_test, model.predict(X_test)))
    print()


def pickle_operating(fname, item):
    # save or load the pickle file.
    file_name = '%s.pickle' % fname
    print(file_name)
    if not item:
        with open(file_name, 'rb') as fs:
            item = pickle.load(fs)
            return item
    else:
        with open(file_name, 'wb') as fs:
            pickle.dump(item, fs, protocol=pickle.HIGHEST_PROTOCOL)


col_names = ['population', 'med_age', '%_bachelor', 'unemploy_rate', 'per_capital',
             'total_households', 'avg_house_size', '%owner_occ_housing', '%renter_occ_housing',
             '%vacant_housing', 'med_home_value', 'population_growth', 'household_growth',
             'per_capital_income_growth', 'winner']

if __name__ == "__main__":
    train_path = './sentences.csv'
    X_train, y_train = load_data(train_path, col_names, False)
    features_sum, norm_train_df = normalize_data(X_train)
    naive_clf = learner('svm', False)
    pca_feature_selection = pca(norm_train_df, 6)
    X_t = pca_feature_selection.transform(norm_train_df)
    tuning_goal = "Macro_F"
    tuned_params = run_tuning_SVM(X_t, np.array(y_train), tuning_goal)[0][0]
    tune_clf = svm.SVC(kernel=tuned_params['kernel'], C=tuned_params['C'], coef0=tuned_params['coef0'],
                       gamma=tuned_params['gamma'], random_state=tuned_params['random_state'], probability=True)
    # run naive model
    experiment(X_train, y_train, naive_clf)
    # run naive model after normalizing data
    experiment(norm_train_df, y_train, naive_clf)
    # run naive model after normalizing data and features reduction
    experiment(X_t, y_train, naive_clf)
    # run tuned model after normalizing data and features reduction
    experiment(X_t, y_train, tune_clf)
    tune_clf.fit(X_t, y_train)
    tuned_model = {'model': tune_clf, 'pca': pca_feature_selection}
    pickle_operating('potus_pred_model', tuned_model)