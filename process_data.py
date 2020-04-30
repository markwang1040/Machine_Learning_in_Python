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
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

def process_data(df, cat_var, cont_var, as_is):
    # feature selection
    class FeatureSelector(BaseEstimator, TransformerMixin):
        def __init__(self, feature):
            self.feature = feature
        def fit(self, X):
            return self
        def transform(self, X):
            return X[self.feature]


    # continuous variables
    if len(cont_var) != 0:
        cont_trans = Pipeline([
            ('FeatureSelector', FeatureSelector(cont_var)),
            ('std', StandardScaler())
        ])
    else:
        cont_trans = Pipeline([
            ('FeatureSelector', FeatureSelector(cont_var)),
            ('Identity', FunctionTransformer())
        ])

    # categorical variables
    if len(cat_var) != 0:
        cat_trans = Pipeline([
            ('FeatureSelector', FeatureSelector(cat_var)),
            ('OneHotEncoding', OneHotEncoder(drop='first', sparse=False))
        ])
    else:
        cat_trans = Pipeline([
            ('FeatureSelector', FeatureSelector(cat_var)),
            ('Identity', FunctionTransformer())
        ])

    # variables to be left alone
    as_is_select = Pipeline([
        ('FeatureSelector', FeatureSelector(as_is)),
        ('Identity', FunctionTransformer())
    ])

    preprocess = FeatureUnion([
        ('cat', cat_trans),
        ('cont', cont_trans),
        ('as_is', as_is_select)
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

    col_names = cat_names + cont_var + as_is
    # collated column names for the df

    return pd.DataFrame(df, columns=col_names)

