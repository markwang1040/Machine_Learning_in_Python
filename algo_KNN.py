import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


class KNN:
    def __init__(self, k_neighbors=3, distance='euclidean'):
        self.k_neighbors = k_neighbors
        self.distance = distance
        assert self.k_neighbors % 1 == 0, "k_neighbors has to be a positive odd integer."
        assert self.k_neighbors > 0, "k_neighbors has to be a positive odd integer."
        assert self.k_neighbors % 2 == 1, "k_neighbors has to be a positive odd integer."
        assert self.distance in ['euclidean',
                                 'manhattan'], "The distance metric has to be either 'euclidean' or 'manhattan'."

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        if type(X_train) == pd.DataFrame:
            X_train = X_train.to_numpy()
        if type(y_train) == pd.DataFrame:
            y_train = y_train.to_numpy()
        assert X_train.shape[0] == y_train.shape[0], "The lengths of X_train and y_train must be equal."
        assert type(X_train) is np.ndarray, "X_train has to be either a Pandas DataFrame or a NumPy array."
        assert type(y_train) is np.ndarray, "y_train has to be either a Pandas DataFrame or a NumPy array."
        self.training_matrix_ = pd.DataFrame((np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)))
        return self

    def nearest_neighbors(self, X):
        if self.distance == 'euclidean':
            self.proximity_matrix_ = pd.DataFrame(pairwise_distances(X, metric='euclidean'))
        elif self.distance == 'manhattan':
            self.proximity_matrix_ = pd.DataFrame(pairwise_distances(X, metric='manhattan'))
        self.proximity_matrix_ = self.proximity_matrix_.replace(0, np.nan)
        self.nearest_neighbors_ = pd.DataFrame({'Index of Observations': self.proximity_matrix_.index.tolist(),
                                                'Index of NN': self.proximity_matrix_.idxmin()}).set_index(
            'Index of Observations')

        return self.nearest_neighbors_

    def predict(self, X_test, objective='reg'):
        self.X_test = X_test
        assert self.X_test.shape[1] == self.X_train.shape[
            1], "The number of columns of X_train and X_test must be equal."
        assert objective in ['reg', 'clf'], "The objective argument must be either 'reg' or 'clf'"
        if self.distance == 'euclidean':
            self.dist_matrix_ = pd.DataFrame(pairwise_distances(self.X_train, self.X_test, metric='euclidean'))
        elif self.distance == 'manhattan':
            self.dist_matrix_ = pd.DataFrame(pairwise_distances(self.X_train, self.X_test, metric='manhattan'))

        self.neighbors_indices_ = pd.DataFrame()

        for col in self.dist_matrix_.columns:
            self.neighbors_indices_ = self.neighbors_indices_.append(
                [self.dist_matrix_[col].nsmallest(self.k_neighbors).index.tolist()])
        self.neighbors_indices_ = self.neighbors_indices_.reset_index().drop(['index'], axis=1)

        neighbors_values_ = self.neighbors_indices_.copy()
        neighbors_values_ = neighbors_values_.applymap(lambda x: knn.training_matrix_.iloc[x, -1])

        if objective == 'reg':
            return pd.DataFrame(neighbors_values_.mean(axis=1), columns=['Predictions'])

        elif objective == 'clf':
            return neighbors_values_.mode(axis=1).rename(columns={0: 'Predictions'})




