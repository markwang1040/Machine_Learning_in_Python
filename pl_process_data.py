# standard imports
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib as plt

# build transformation pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion


# Authors: Mark Wang <markswang@uchicago.edu>


def process_data(df, cat_var, cont_var):
    # feature selection
    class FeatureSelector(BaseEstimator, TransformerMixin):
        def __init__(self, feature):
            self.feature = feature
        def fit(self, X):
            return self
        def transform(self, X):
            return X[self.feature]

    # continuous variables
    cont_trans = Pipeline([
        ('FeatureSelector', FeatureSelector(cont_var)),
        ('std', StandardScaler())
    ])

    # categorical variables
    cat_trans = Pipeline([
        ('FeatureSelector', FeatureSelector(cat_var)),
        ('OneHotEncoding', OneHotEncoder(drop='first', sparse=False))
    ])

    preprocess = FeatureUnion([
        ('cat', cat_trans),
        ('cont', cont_trans)
    ])

    # store the transformed df in a variable
    # the purpose is to fit the OneHotEncoders first, so that the column names can be extracted for use below
    df = preprocess.fit_transform(df)

    cat_names = []
    for i in range(len(cat_trans[1].categories_)): # iterate through the list of arrays of levels of each column
        list_of_col_names = cat_trans[1].categories_[i].tolist()[1:]
        # since the first level is dropped in encoding, the list of column names should start from [1]
        list_of_col_names = [cat_var[i] + '_' + str(item) for item in list_of_col_names]
        # parent column's name + '_' + level of column
        cat_names += list_of_col_names
        # create a list of collated column names for categorical variables

    col_names = cat_names + cont_var
    # collated column names for the df

    return pd.DataFrame(df, columns=col_names)

