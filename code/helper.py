from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

import pandas as pd
import sys

sys.dont_write_bytecode = True

HEADER = 'dob_yy,dob_mm,dob_wk,ostate,ubfacil,mager9,metrores,rectype,restatus,umhisp,mar,meduc_rec,fracerec,ufhisp,lbo_rec,tbo_rec,mpcb_rec6,wtgain_rec,u_apncu,tobuse,cig_rec6,alcohol,drinks_rec,urf_anemia,urf_card,urf_lung,urf_diab,urf_gen,urf_hydra,urf_hemo,urf_chyper,urf_phyper,urf_eclam,urf_incerv,urf_pre4000,urf_preterm,urf_renal,urf_rh,urf_uterine,urf_other,uop_amnio,uop_monit,uop_induc,uop_stiml,uop_tocol,uop_ultra,uop_other,uld_febr,uld_meco,uld_ruptr,uld_abrup,uld_prepla,uld_excbl,uld_seiz,uld_precip,uld_prolg,uld_dysfn,uld_breech,uld_cephal,uld_cord,uld_anest,uld_distr,uld_other,ume_vag,ume_vbac,ume_primc,ume_repec,ume_forcp,ume_vac,attend,apgar5r,dplural,gestrec10,bwtr14,uab_anem,uab_injury,uab_alcoh,uab_hyal,uab_mecon,uab_venl30,uab_ven30m,uab_nseiz,uab_other,aged,female'


def split_train_test(filedir, split_threshold=0.2):
    """
    Code to split the binary dataset into training and testing sets.
    The labels are: mortal class (assigned by -1),
                    survival class (assigned by 1).

    :param filedir (string): The filename of the dataset.
    :param split_threshold (float): The threshold that refers to the testing set.

    :return: tuple: A tuple containing the train and test features lists, and the train and test labels lists.

    """
    
    df = pd.read_csv(filedir, names=HEADER.split(','))

    df.loc[df['aged'] >= 0, 'aged'] = -1
    df.loc[df['aged'] != -1, 'aged'] = 1
    df_labels = df[['aged']]

    df = df.drop(['aged'], axis=1)

    return train_test_split(df, df_labels, test_size=split_threshold)


def change_minority_class_label_to_zero(df_labels):
    """
    Changes the labels of the minority class from -1 to 0.
    The labels are: mortal class (assigned by 0),
                    survival class (assigned by 1).

    :param df_labels: pandas data frame containing the labels.

    :return: pandas data frame: containing the converted valued labels.

    """

    df_labels.loc[df_labels['aged'] == -1, 'aged'] = 0

    return df_labels[['aged']]


def split_train_test_multi(filedir, split_threshold=0.2):
    """
    Code to split the multilabel dataset into training and testing sets.
    The labels are: survival class (assigned by 0),
                    perinatal class (assigned by 1),
                    neonatal class (assigned by 2),
                    postneonatal class (assigned by 3).

    :param filedir (string): The filename of the dataset.
    :param split_threshold (float): The threshold that refers to the testing set.

    :return: tuple: A tuple containing the train and test features lists, and the train and test labels lists.

    """

    df = pd.read_csv(filedir, names=HEADER.split(','))

    df_labels = df[['aged']]

    df = df.drop(['aged'], axis=1)

    return train_test_split(df, df_labels, test_size=split_threshold)


def split_train_test_multi_binarize(fileDir, split_threshold=0.2, class_list=[0, 1, 2, 3]):
    """
    Code to split the multilabel dataset into training and testing sets.
    The labels are: in the first column is the survival class (assigned by 1), otherwise 0
                    in the first column is the perinatal class (assigned by 1), otherwise 0
                    in the first column is the neonatal class (assigned by 1), otherwise 0
                    in the first column is the postneonatal class (assigned by 1), otherwise 0.

    :param filedir (string): The filename of the dataset.
    :param split_threshold (float): The threshold that refers to the testing set.
    :param class_list (integer list): The ids of the classes.

    :return: tuple: A tuple containing the train and test features lists, and the train and test labels lists.

    """

    df = pd.read_csv(fileDir, names = HEADER.split(','))

    df_labels = df[['aged']]

    df = df.drop(['aged'], axis=1)

    labels = label_binarize(df_labels, classes=class_list)

    return train_test_split(df, labels, test_size=split_threshold)


def evaluate(test_labels, preds, n_decimals=2, binary=True):
    """
    Code to evaluate the predictions of our model.

    :param test_labels (list): The labels list of our testing set.
    :param preds (list): The predicitions (given by our model) of our testing set.
    :param n_decimals (float): The number of decimals to keep per metric.
    :param binary (boolean): Whether the evaluation is done for the binary or the multiclass task.

    :return: list: A list containing the evaluation metrics [accuracy, precision, recall, f-score, area under the curve].

    """

    accuracy = round(accuracy_score(test_labels, preds), n_decimals)

    # [evaluation on minority class, evaluation on majority class]
    prec_list, rec_list, f_list, _ = precision_recall_fscore_support(test_labels, preds, pos_label=1)
    precision = [round(p, n_decimals) for p in prec_list]
    recall = [round(r, n_decimals) for r in rec_list]
    f_score = [round(f, n_decimals) for f in f_list]

    metrics_list = [accuracy, precision, recall, f_score]

    if binary:
        auc = round(roc_auc_score(test_labels, preds), n_decimals)
        metrics_list.append(auc)

    return metrics_list


def print_evaluation(evaluation_metrics, binary=True):
    """
    Code to evaluate the predictions of our model.

    :param evaluation_metrics (list): The evaluation metrics list.
    :param binary (boolean): Whether the evaluation is done for the binary or the multiclass task.

    :return: string: A string containing the evaluation metrics for the minority class.

    """

    evaluation_metrics_minority = []
    for s in evaluation_metrics:
        if binary and type(s) is list:
            evaluation_metrics_minority.append(str(s[0]))
        else:
            evaluation_metrics_minority.append(str(s))

    return '\t'.join(evaluation_metrics_minority) + '\t'