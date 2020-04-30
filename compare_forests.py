import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

# Author: Mark Wang <markswang@uchicago.edu>

'''
Compare changes in feature importances between two random forests, 
presumably learnt from the same data with the only column different being the target variable, 
to reflect changes in different metrics based on the same sample, or changes in a metric due to passage of time.
'''

class CompareForests:



    def __init__(self):
        pass



    @staticmethod
    def longestCommonPrefix(m):
        if not m: return ''

        # since list of string will be sorted and retrieved min max by alphebetic order
        s1 = min(m)
        s2 = max(m)

        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]  # stop until hit the split index
        return s1



    def importance_changes(self, model1, model2):
        assert ('feature_names_before_' and 'feature_names_after_') in dir(model1), \
            "'feature_names_before_' and 'feature_names_after' not found in model1 object."
        assert ('feature_names_before_' and 'feature_names_after_') in dir(model2), \
            "'feature_names_before_' and 'feature_names_after' not found in model2 object."
        assert type(model1) == sklearn.ensemble._forest.RandomForestClassifier, \
            "Model1 is of an unspportued type."
        assert type(model2) == sklearn.ensemble._forest.RandomForestClassifier, \
            "Model2 is of an unspportued type."
        assert model1.feature_names_before_ == model2.feature_names_before_, \
                                              "The two models have different feature names before data is processed."
        assert model1.feature_names_after_ == model2.feature_names_after_, \
                                             "The two models have different feature names after data is processed."

        self.model1 = model1
        self.model2 = model2


        f_imp1 = model1.feature_importances_
        f_imp2 = model2.feature_importances_

        f_imp_scattered_dict_1 = {key:value for key, value in zip(model1.feature_names_after_, f_imp1)}
        f_imp_scattered_dict_2 = {key:value for key, value in zip(model2.feature_names_after_, f_imp2)}

        f_imp_collated_dict_1 = {key:0 for key in model1.feature_names_before_}
        f_imp_collated_dict_2 = {key:0 for key in model2.feature_names_before_}

        for key1 in f_imp_scattered_dict_1.keys():
            checklist = []
            for key2 in f_imp_collated_dict_1.keys():

                if key1.startswith(key2):
                    checklist.append(key2)

            if len(checklist) == 1:
                f_imp_collated_dict_1[checklist[0]] += f_imp_scattered_dict_1[key1]

            else:
                prefix_list = []

                for item in checklist:
                    prefix = self.longestCommonPrefix([item, key2])
                    prefix_list.append(prefix)

                idx = max(enumerate(prefix_list), key=lambda x: len(x[1]))
                f_imp_collated_dict_1[checklist[idx[0]]] += f_imp_scattered_dict_1[key1]

        for key1 in f_imp_scattered_dict_2.keys():
            checklist = []
            for key2 in f_imp_collated_dict_2.keys():

                if key1.startswith(key2):
                    checklist.append(key2)

            if len(checklist) == 1:
                f_imp_collated_dict_2[checklist[0]] += f_imp_scattered_dict_2[key1]

            else:
                prefix_list = []

                for item in checklist:
                    prefix = self.longestCommonPrefix([item, key2])
                    prefix_list.append(prefix)

                idx = max(enumerate(prefix_list), key=lambda x: len(x[1]))
                f_imp_collated_dict_2[checklist[idx[0]]] += f_imp_scattered_dict_2[key1]


        results = pd.DataFrame.from_dict(data=f_imp_collated_dict_1, orient='index', columns=['Feature Importance Model1'])\
            .join(pd.DataFrame.from_dict(data=f_imp_collated_dict_2, orient='index', columns=['Feature Importance Model2']))

        results['Importace Change'] = results['Feature Importance Model2'] - results['Feature Importance Model1']
        results['% Importance Change'] = results['Importace Change'] / results['Feature Importance Model1']
        results['Rank Model1'] = results['Feature Importance Model1'].rank(method='min', ascending=False)
        results['Rank Model2'] = results['Feature Importance Model2'].rank(method='min', ascending=False)
        results['Rank Change'] = results['Rank Model1'] - results['Rank Model2']
        results = results.sort_values('Rank Model1')

        self.results_ = results

        return results














