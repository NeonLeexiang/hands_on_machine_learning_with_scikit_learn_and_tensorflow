"""
    date:       2019/12/30 8:37 下午
    written by: neonleexiang
"""


# import list
import fetch_mnist_data
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# inherit the methods
get_data = fetch_mnist_data.load_data_from_datasets


def train_test_split(X, y):
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    return X_train, X_test, y_train, y_test


# SGD Classifier
def SGD_Classifier(X_data, y_data, _max_iter=5, _tol=-np.infty, _random_state=42):
    sgd_clf = SGDClassifier(max_iter=_max_iter, tol=_tol, random_state=_random_state)

    # scaler = StandardScaler()
    # X_data_scaled = scaler.fit_transform(X_data, astype(np.float64))

    sgd_clf.fit(X_data, y_data)

    return sgd_clf


# randomForest Classifier
def RF_Classifier(X_data, y_data, _n_estimators=10, _random_state=42):
    forest_clf = RandomForestClassifier(n_estimators=_n_estimators, random_state=_random_state)

    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data.astype(np.float64))

    forest_clf.fit(X_data_scaled, y_data)

    return forest_clf


def RF_clf_confusion_matrix(RF_clf, X_data, y_data, _cv=3):
    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data.astype(np.float64))

    c_v_score = cross_val_score(RF_clf, X_data_scaled, y_data, cv=_cv, scoring='accuracy')

    print("cross val score is : \n", c_v_score)

    y_pred = cross_val_predict(RF_clf, X_data_scaled, y_data, cv=_cv)

    conf_matrix = confusion_matrix(y_data, y_pred)

    print(conf_matrix)

    return conf_matrix


def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.matshow(matrix, cmap=plt.cm.gray)

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_mx = matrix / row_sums

    np.fill_diagonal(norm_mx, 0)
    plt.matshow(norm_mx, cmap=plt.cm.gray)

    plt.show()


# store the model
def store_model(training_model, name):
    name += '.pkl'
    joblib.dump(training_model, name)


if __name__ == '__main__':
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # SGD Classifier
    # sgd_classifier = SGD_Classifier(X_train, y_train)

    # forest_clf = RF_Classifier(X_train, y_train)
    # forest_conf_mx = RF_clf_confusion_matrix(forest_clf, X_train, y_train)
    # plot_confusion_matrix(forest_conf_mx)
    #
    # store_model(forest_clf, 'RandomForestClassifier')

    final_model = joblib.load("RandomForestClassifier.pkl")
    final_conf_mx = RF_clf_confusion_matrix(final_model, X_train, y_train)
    plot_confusion_matrix(final_conf_mx)





