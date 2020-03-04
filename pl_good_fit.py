import pandas as pd
import numpy as np
from sklearn.metrics \
    import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

# Authors: Mark Wang <markswang@uchicago.edu>,
#          Kan Liu <liukan07@berkeley.edu>.

class GoodFit:
    def __init__(self, objective):
        assert objective in ['reg', 'clf'], "The objective has to be either 'reg' or 'clf'."
        # if objective == 'reg':
        #     assert set(criteria).issubset(['mse', 'mae']), \
        #         "The criteria for regression models has to be from the list ['mse', 'mae']."
        # else:
        #     assert set(criteria).issubset(['acc', 'recall', 'precision', 'f1']), \
        #         "The criteria for regression models has to be from the list ['acc', 'recall', 'precision', 'f1']."

        self.objective = objective
        # self.criteria = criteria

        print("GoodFit(objective={})".format(self.objective))


    def find_the_spot(self, model, params_grid, X_train, y_train, X_test, y_test):
        assert type(params_grid) is dict, "The params_grid needs to a dictionary in which the keys are the possible " \
                                          "parameters of the model, and the values are lists of possible instances of " \
                                          "the parameters."
        assert set(params_grid.keys()).issubset(list(model.get_params().keys())), "Parameter in params_grid not " \
                                                                                  "applicable to current model."
        for value in params_grid.values():
            assert type(value) is list, "Values in params_grid needs to be lists of possible instances of each parameter."

        assert (type(X_train) is np.ndarray) | (type(X_train) is pd.DataFrame) | (type(X_train) is pd.Series), \
            "Data needs to be either a Pandas DataFrame, Series or a Numpy array."
        assert (type(y_train) is np.ndarray) | (type(y_train) is pd.DataFrame)| (type(y_train) is pd.Series), \
            "Data needs to be either a Pandas DataFrame, Series or a Numpy array."
        assert (type(X_test) is np.ndarray) | (type(X_test) is pd.DataFrame)| (type(X_test) is pd.Series), \
            "Data needs to be either a Pandas DataFrame, Series or a Numpy array."
        assert (type(y_test) is np.ndarray) | (type(y_test) is pd.DataFrame)| (type(y_test) is pd.Series), \
            "Data needs to be either a Pandas DataFrame, Series or a Numpy array."

        assert X_test.shape[1] == X_train.shape[1], "The number of columns of X_train and X_test must be equal."

        self.model = model
        self.params_grid = params_grid

        list_of_param_combinations = ParameterGrid(params_grid)
        results = pd.DataFrame(list_of_param_combinations)

        if self.objective == 'reg':
            train_mse = []
            test_mse = []
            train_mae = []
            test_mae = []
            estimator = []
            for combo in list_of_param_combinations:
                model = model.set_params(**combo)
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                mae_train = mean_absolute_error(y_train, y_pred_train)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                train_mse.append(mse_train)
                test_mse.append(mse_test)
                train_mae.append(mae_train)
                test_mae.append(mae_test)
                estimator.append(model)
            MSE_diff = [i - j for i, j in zip(train_mse, test_mse)]
            MAE_diff = [i - j for i, j in zip(train_mae, test_mae)]
            self.estimators_ = estimator
            scores = pd.DataFrame({'MSE_train':train_mse, 'MSE_test':test_mse, 'MSE_diff':MSE_diff,
                               'MAE_train':train_mae, 'MAE_test':test_mae, 'MAE_diff':MAE_diff})
            self.results_ = results.join(scores)
            return self.results_

        else:
            pass





