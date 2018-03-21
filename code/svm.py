from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

import sys

sys.dont_write_bytecode = True


def predict_with_one_class_SVM(train, test, train_labels, test_labels):
    """
    Code to run a 1-class SVMs classifier for binary classification

    :param train (list): The features of the train data.
    :param test (list): The features of the test data.
    :param train_labels (list): The labels of the train data.
    :param test_labels (list): The labels of the test data.

    :return: A binary list containing the predictions.

    """

    train_died = (train_labels['aged'] == -1.0).sum()
    outliers = train_died

    train_lived = (train_labels['aged'] == 1.0).sum()
    all_dp = train_lived + outliers

    # Set nu (which should be the proportion of outliers in our dataset)
    nu = float(outliers) / all_dp

    # Initialize the classifier
    model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.000000001)

    # Train the classifier
    model = model.fit(train, train_labels)

    # Make predictions
    preds = model.predict(test)

    return preds


def predict_with_one_class_SVM_multi(train, test, train_labels, test_labels):
    """
    Code to run a one-class SVMs classifier for multi label classification.

    :param train (list): The features of the train data.
    :param test (list): The features of the test data.
    :param train_labels (list): The labels of the train data.
    :param test_labels (list): The labels of the test data.

    :return: A binary list containing the predictions.

    """
    """
    train_died = (train_labels['aged'] == -1.0).sum()
    outliers = train_died

    train_lived = (train_labels['aged'] == 1.0).sum()
    all_dp = train_lived + outliers

    # Set nu (which should be the proportion of outliers in our dataset)
    nu = float(outliers) / all_dp
    
    # Initialize the classifier
    model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.00001)
    """

    # Initialize the classifier
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))

    # Train the classifier
    model = model.fit(train, train_labels)

    # Make predictions
    preds = model.predict(test)

    return preds

