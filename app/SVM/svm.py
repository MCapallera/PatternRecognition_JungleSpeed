from sklearn import preprocessing
from SVM.feature import to_hog
from SVM.model import test_svm
from util.dataset import get_training_set, get_test_set, Data

train = Data(*get_training_set())
test = Data(*get_test_set())

grid_search_params = [
    {"kernel": ["linear"], "C": [0.1, 1, 5]},
    {"kernel": ["rbf"], "C": [0.1, 1, 5], "gamma": [0.001, 0.01, 0.05]}
]

optimal_params = {
    "kernel": "rbf", "C": 1, "gamma": 0.5
}

feature_extraction_mods = ['plain', 'hog']
# feature_extraction_mods = ['hog']

for feature_extraction_mod in feature_extraction_mods:
    if feature_extraction_mod == 'hog':
        working_train = to_hog(train)
        working_test = to_hog(test)
        scaler = preprocessing.StandardScaler().fit(working_train.x)
        working_train.x = scaler.transform(working_train.x)
        working_test.x = scaler.transform(working_test.x)
    else:
        working_train = train
        working_test = test

    test_svm(grid_search_params, working_train, working_test)







