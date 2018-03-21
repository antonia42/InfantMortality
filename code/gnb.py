from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
import sys

sys.dont_write_bytecode = True


def predict_with_GNB(train, test, train_labels, test_labels):
    """
    Code to run a Gaussian Naive Bayes classifier for binary classification.

    :param train (list): The features of the train data.
    :param test (list): The features of the test data.
    :param train_labels (list): The labels of the train data.
    :param test_labels (list): The labels of the test data.

    :return: A binary list containing the predictions.

    """

    # Initialize the classifier
    gnb = GaussianNB()

    # Train the classifier
    gnb = gnb.fit(train, train_labels.values.ravel())

    # Make predictions
    preds = gnb.predict(test)

    return preds


def predict_with_GNB_multi(train, test, train_labels, test_labels):
    """
    Code to run a Gaussian Naive Bayes classifier for multi label classification.

    :param train (list): The features of the train data.
    :param test (list): The features of the test data.
    :param train_labels (list): The labels of the train data.
    :param test_labels (list): The labels of the test data.

    :return: A binary list containing the predictions.

    """

    # Initialize the classifier
    gnb = OneVsRestClassifier(GaussianNB())

    # Train the classifier
    gnb = gnb.fit(train, train_labels)

    # Make predictions
    preds = gnb.predict(test)

    return preds

