import sys
import time

from xgb import predict_with_XGB, predict_with_XGB_multi
from gnb import predict_with_GNB, predict_with_GNB_multi
from helper import split_train_test, split_train_test_multi, split_train_test_multi_binarize, change_minority_class_label_to_zero, evaluate, print_evaluation
from svm import predict_with_one_class_SVM, predict_with_one_class_SVM_multi

sys.dont_write_bytecode = True


DATASETDIR = './../dataset/'
split_threshold = 0.3
n_decimals = 2
secs = 60.0


def binary_prediction(filedir, model):
    """
    Runs and evaluates for the requested model for the binary classification.

    :param filedir (string): The filename of the dataset.
    :param model (string): The requested model to run.

    :return: string containing the evaluation metrics values.

    """

    train, test, train_labels, test_labels = split_train_test(filedir, split_threshold)

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
        for model in models:
            eval_metrics = binary_prediction(DATASETDIR + fn, model)
            print model + '\t' + eval_metrics



def multi_prediction(filedir, model):
    """
    Runs and evaluates for the requested model for the multi label classification.

    :param filedir (string): The filename of the dataset.
    :param model (string): The requested model to run.

    :return: string containing the evaluation metrics values.

    """

    if model == 'SVM':
        train, test, train_labels, test_labels = split_train_test_multi_binarize(filedir, split_threshold)
        t0 = time.time()
        preds = predict_with_one_class_SVM_multi(train, test, train_labels, test_labels)

    elif model == 'GNB':
        train, test, train_labels, test_labels = split_train_test_multi_binarize(filedir, split_threshold)
        t0 = time.time()
        preds = predict_with_GNB_multi(train, test, train_labels, test_labels)

    elif model == 'XGB':
        train, test, train_labels, test_labels = split_train_test_multi(filedir, split_threshold)
        t0 = time.time()
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
    print 'Method\tAcc\tPrec (SURV,PERI,NEO,POST)\tRec (SURV,PERI,NEO,POST)\tF (SURV,PERI,NEO,POST)\tRuntime'

    for fn in filenames:
        for model in models:
            eval_metrics = multi_prediction(DATASETDIR + fn, model)
            print model + '\t' + eval_metrics


if __name__ == '__main__':
    models = ['GNB', 'XGB', 'SVM']

    filenames_binary = ['1_1_mean.csv', '1_5_mean.csv']
    binary_main(filenames_binary, models)

    filenames_multi = ['1_1_mean_multi.csv', '1_5_mean_multi.csv']
    multi_main(filenames_multi, models)

