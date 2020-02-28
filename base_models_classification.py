import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import datetime as dt


def fit_classification_models(X_train, y_train, X_test, y_test):

    rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=3, oob_score=True)
    gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=500, random_state=3)
    svc = SVC(degree=2, random_state=3)
    xgbc = XGBClassifier(learning_rate=0.1, n_estimators=500, random_state=3, n_jobs=-1)

    regressors = [rfc, gbc, svc, xgbc]

    model = []
    training_acc = []
    testing_acc = []
    training_f1 = []
    testing_f1 = []
    training_recall = []
    testing_recall = []
    training_precision = []
    testing_precicion = []
    time_taken = []

    for i in regressors:
        begin = dt.datetime.now()

        i.fit(X_train, y_train)
        y_pred_train = i.predict(X_train)
        y_pred_test = i.predict(X_test)
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)
        recall_train = recall_score(y_train, y_pred_train)
        recall_test = recall_score(y_test, y_pred_test)
        precision_train = precision_score(y_train, y_pred_train)
        precision_test = precision_score(y_test, y_pred_test)


        run_time = dt.datetime.now() - begin

        model.append(str(i).split('(')[0])
        training_acc.append(acc_train)
        testing_acc.append(acc_test)
        training_f1.append(f1_train)
        testing_f1.append(f1_test)
        training_recall.append(recall_train)
        testing_recall.append(recall_test)
        training_precision.append(precision_train)
        testing_precicion.append(precision_test)

        time_taken.append(run_time)

    results = pd.DataFrame(zip(model, training_acc, testing_acc, training_f1, testing_f1,
                               training_recall, testing_recall, training_precision, testing_precicion,
                               time_taken),
                           columns = ['model', 'trainint_acc', 'testing_acc', 'training_f1', 'testing_f1',
                                      'training_recall', 'testing_recall', 'training_precision', 'testing_precicion',
                                      'time_taken'])

    return(rfc, gbc, svc, xgbc, results)


