import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

import datetime as dt


def fit_regression_models(X_train, y_train, X_test, y_test):

    rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=3, oob_score=True)
    gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=3)
    en = ElasticNet(l1_ratio=0.5, max_iter=100000, random_state=3)
    xgbr = XGBRegressor(learning_rate=0.1, n_estimators=500, random_state=3, n_jobs=-1)

    regressors = [rfr, gbr, en, xgbr]

    model = []
    training_error = []
    testing_error = []
    time_taken = []

    for i in regressors:
        begin = dt.datetime.now()

        i.fit(X_train, y_train)
        y_pred_train = i.predict(X_train)
        y_pred_test = i.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        run_time = dt.datetime.now() - begin

        model.append(str(i).split('(')[0])
        training_error.append(mse_train)
        testing_error.append(mse_test)
        time_taken.append(run_time)

    results = pd.DataFrame(zip(model, training_error, testing_error, time_taken),
                           columns = ['model', 'trainint_error', 'testing_error', 'time_taken'])

    return(rfr, gbr, en, xgbr, results)


