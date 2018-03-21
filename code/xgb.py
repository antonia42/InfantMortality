import sys
#from xgboost import XGBClassifier
import xgboost as XGBClassifier
from helper import change_minority_class_label_to_zero

sys.dont_write_bytecode = True


def predict_with_XGB(train, test, train_labels, test_labels):

    # Grid Search CV optimized settings
    obj_binary = 'binary:logistic'
    best_nround_binary = 889
    optimized_params_binary = {'eta': 0.1, 'seed': 0, 'subsample': 0.7, 'learning_rate': 0.01,
                         'n_estimators': 1250, 'colsample_bytree': 0.7, 'objective': obj_binary,
                         'max_depth': 3, 'min_child_weight': 3}

    # Initialize the classifier DMatrix to make XGBoost more efficient
    xgdmat = XGBClassifier.DMatrix(train, train_labels)

    # Train the classifier
    final_gb = XGBClassifier.train(optimized_params_binary, xgdmat, num_boost_round=best_nround_binary)
    testdmat = XGBClassifier.DMatrix(test)

    # Make predictions
    y_pred = final_gb.predict(testdmat)  # Predict using testdmat

    preds = [round(value) for value in y_pred]

    return preds


def predict_with_XGB_multi(train, test, train_labels, test_labels):

    # Grid Search CV optimized settings
    obj_multi = 'multi:softmax'
    best_nround_multi = 1216
    optimized_params_multi = {'eta': 0.1, 'seed': 0, 'subsample': 0.7, 'learning_rate': 0.01,
                        'n_estimators': 1250, 'colsample_bytree': 0.8, 'objective': obj_multi,
                        'max_depth': 3, 'min_child_weight': 3, 'num_class': 4}

    # Initialize the classifier DMatrix to make XGBoost more efficient
    xgdmat = XGBClassifier.DMatrix(train, train_labels)

    # Train the classifier
    final_gb = XGBClassifier.train(optimized_params_multi, xgdmat, num_boost_round=best_nround_multi)
    testdmat = XGBClassifier.DMatrix(test)

    # Make predictions
    y_pred = final_gb.predict(testdmat)  # Predict using testdmat

    preds = [round(value) for value in y_pred]

    return preds

