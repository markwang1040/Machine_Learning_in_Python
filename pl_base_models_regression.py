import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime as dt


def fit_regression_models(X_train, y_train, X_test, y_test, random_state):

    rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=random_state, oob_score=True)
    gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=random_state)
    en = ElasticNet(l1_ratio=0.5, max_iter=100000, random_state=random_state)
    xgbr = XGBRegressor(learning_rate=0.1, n_estimators=500, random_state=random_state, n_jobs=-1)

    regressors = [rfr, gbr, en, xgbr]

    model = []
    training_error_mse = []
    testing_error_mse = []
    training_error_mae = []
    testing_error_mae = []
    time_taken = []

    for i in regressors:
        begin = dt.datetime.now()

        i.fit(X_train, y_train)
        y_pred_train = i.predict(X_train)
        y_pred_test = i.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)


        run_time = dt.datetime.now() - begin

        model.append(str(i).split('(')[0])
        training_error_mse.append(mse_train)
        testing_error_mse.append(mse_test)
        training_error_mae.append(mae_train)
        testing_error_mae.append(mae_test)

        time_taken.append(run_time)

    results = pd.DataFrame(zip(model, training_error_mse, testing_error_mse, training_error_mae, testing_error_mae,
                               time_taken), columns = ['model', 'trainint_error_mse', 'testing_error_mse',
                                                       'trainint_error_mae', 'testing_error_mae','time_taken'])

    return(rfr, gbr, en, xgbr, results)


