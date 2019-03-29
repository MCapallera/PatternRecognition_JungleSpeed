from sklearn import svm, utils
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from util.dataset import Data


def test_svm(grid_search_param, train:Data, test:Data):
    # request probability estimation
    model = svm.SVC(probability=True)

    # change backend to threading as loky fails to serialize (and hope they removed GIL -> looking at the performance tool it seems so)
    utils.parallel_backend('threading')

    # 10-fold cross validation, use some threads as each fold and each parameter set can be train in parallel
    clf = GridSearchCV(model, grid_search_param, cv=10, n_jobs=8, verbose=3)
    clf.fit(train.x, train.y)

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict = clf.predict(test.x)

    print("\nConfusion matrix:")
    print(confusion_matrix(test.y, y_predict))

    print("\nClassification report:")
    print(classification_report(test.y, y_predict))

