import sys
import time

from xgb import predict_with_XGB, predict_with_XGB_multi
from gnb import predict_with_GNB, predict_with_GNB_multi
from helper import split_train_test, split_train_test_multi_binarize, change_minority_class_label_to_zero, create_one_col_for_labels, evaluate, print_evaluation
from svm import predict_with_one_class_SVM, predict_with_one_class_SVM_multi

sys.dont_write_bytecode = True


DATASETDIR = './../dataset/'
split_threshold = 0.3
n_decimals = 2
secs = 60.0


def binary_prediction(train, test, train_labels, test_labels, model):
    """
    Runs and evaluates for the requested model for the binary classification.

    :param train (pandas data frame): The training set of the data containing the features.
    :param test (pandas data frame): The testing set of the data containing the features.
    :param train_labels (pandas data frame): The training set of the data containing the labels.
    :param test_labels (pandas data frame): The testing set of the data containing the labels.
    :param model (string): The requested model to run.

    :return: string containing the evaluation metrics values.

    """

    t0 = time.time()

    if model == 'SVM':
        preds = predict_with_one_class_SVM(train, test, train_labels, test_labels)

    elif model == 'GNB':
        preds = predict_with_GNB(train, test, train_labels, test_labels)

    elif model == 'XGB':
        train_labels = change_minority_class_label_to_zero(train_labels)
        test_labels = change_minority_class_label_to_zero(test_labels)
        preds = predict_with_XGB(train, test, train_labels, test_labels)

    t1 = time.time()

    if len(preds) > 0:
        class_metrics = evaluate(test_labels, preds, n_decimals)
        evaluation = print_evaluation(class_metrics)

    runtime = str(round((t1 - t0) / secs, n_decimals))

    return evaluation + runtime


def binary_main(filenames, models):
    """
    Runs the binary main for all filenames and all models.

    :param filenames (list of strings): The datasets.
    :param model (list of strings): The models to run.

    :return: nothing, prints the evaluation metrics values.

    """

    print '\t\tBINARY CLASSIFICATION'
    print 'Method\tAcc\tPrec\tRec\tF\tAUC\tRuntime'

    for fn in filenames:
        train, test, train_labels, test_labels = split_train_test(DATASETDIR + fn, split_threshold)

        for model in models:
            eval_metrics = binary_prediction(train, test, train_labels, test_labels, model)
            print model + '\t' + eval_metrics


def multi_prediction(train, test, train_labels, test_labels, model):
    """
    Runs and evaluates for the requested model for the multi label classification.

    :param train (pandas data frame): The training set of the data containing the features.
    :param test (pandas data frame): The testing set of the data containing the features.
    :param train_labels (pandas data frame): The training set of the data containing the labels.
    :param test_labels (pandas data frame): The testing set of the data containing the labels.
    :param model (string): The requested model to run.

    :return: string containing the evaluation metrics values.

    """

    t0 = time.time()
    if model == 'SVM':
        preds = predict_with_one_class_SVM_multi(train, test, train_labels, test_labels)

    elif model == 'GNB':
        preds = predict_with_GNB_multi(train, test, train_labels, test_labels)

    elif model == 'XGB':
        train_labels = create_one_col_for_labels(train_labels)
        test_labels = create_one_col_for_labels(test_labels)
        preds = predict_with_XGB_multi(train, test, train_labels, test_labels)

    t1 = time.time()

    if len(preds) > 0:
        class_metrics = evaluate(test_labels, preds, n_decimals, False)
        evaluation = print_evaluation(class_metrics, False)

    runtime = str(round((t1 - t0) / secs, n_decimals))

    return evaluation + runtime


def multi_main(filenames, models):
    """
    Runs the multi-label classification main for all filenames and all models.

    :param filenames (list of strings): The datasets.
    :param model (list of strings): The models to run.

    :return: nothing, prints the evaluation metrics values.

    """
    print '\t\tMULTI-LABEL CLASSIFICATION'
    print 'Method\tAcc\tPrec (SURV,PERI,NEO,POST)\tRec (SURV,PERI,NEO,POST)\tF (SURV,PERI,NEO,POST)\t\tRuntime'

    for fn in filenames:
        train, test, train_labels, test_labels = split_train_test_multi_binarize(DATASETDIR + fn, split_threshold)
        for model in models:
            eval_metrics = multi_prediction(train, test, train_labels, test_labels, model)
            print model + '_' + ''.join(fn[:3]) + '\t' + eval_metrics


if __name__ == '__main__':
    models = ['GNB', 'SVM', 'XGB']

    filenames_binary = ['1_1_mean.csv', '1_5_mean.csv']
    binary_main(filenames_binary, models)

    filenames_multi = ['1_1_mean_multi.csv', '1_5_mean_multi.csv']
    multi_main(filenames_multi, models)

