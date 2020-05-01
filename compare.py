import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier



class Woodsmith:


    def __init__(self):

        pass


    def longestCommonPrefix(self, m):

        if not m: return ''

        # making use of the fact that the list of strings will be sorted and items retrieved min/max by alphebetical order
        s1 = min(m)
        s2 = max(m)

        for i, c in enumerate(s1):
            if c != s2[i]:
                # stop until the split index is reached
                return s1[:i]
        return s1


    def importance_changes(self, model1, model2, cutoff=None):
        ###########################
        # the first part outputs the results table, which contains changes in rank, %imp and imp of all features
        # asserts are pretty much self-explanatory
        assert type(model1) == type(model2), "The two models have to be of the same type for comparison."

        assert ('feature_names_before_' and 'feature_names_after_') in dir(model1), \
            "`feature_names_before` and `feature_names_after` not found in model1 object."
        assert ('feature_names_before_' and 'feature_names_after_') in dir(model2), \
            "`feature_names_before_` and `feature_names_after` not found in model2 object."

        assert type(model1) == (sklearn.ensemble._forest.RandomForestClassifier or
                                sklearn.tree._classes.DecisionTreeClassifier or sklearn.tree._classes.ExtraTreeClassifier), \
            "Model1 is of an unspportued type."
        assert type(model2) == (sklearn.ensemble._forest.RandomForestClassifier or
                                sklearn.tree._classes.DecisionTreeClassifier or sklearn.tree._classes.ExtraTreeClassifier), \
            "Model1 is of an unspportued type."

        assert model1.feature_names_before_ == model2.feature_names_before_, \
                                              "The two models have different feature names before data is processed."
        assert model1.feature_names_after_ == model2.feature_names_after_, \
                                             "The two models have different feature names after data is processed."

        # store models for use perhaps with later methods
        self.model1 = model1
        self.model2 = model2

        # extract feature importances
        f_imp1 = model1.feature_importances_
        f_imp2 = model2.feature_importances_

        # construct feature importances dicts, with feature names being those after one-hot encoding
        f_imp_scattered_dict_1 = {key:value for key, value in zip(model1.feature_names_after_, f_imp1)}
        f_imp_scattered_dict_2 = {key:value for key, value in zip(model2.feature_names_after_, f_imp2)}

        # construct feature importances dicts with the original feature names; values initiated at 0
        f_imp_collated_dict_1 = {key:0 for key in model1.feature_names_before_}
        f_imp_collated_dict_2 = {key:0 for key in model2.feature_names_before_}

        # add the scattered feature importances to the dicts of collated feature importances for model1
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

        # and the same thing again for model2
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

        # results dataframe
        all_results = pd.DataFrame.from_dict(data=f_imp_collated_dict_1, orient='index', columns=['Feature Importance Model1'])\
            .join(pd.DataFrame.from_dict(data=f_imp_collated_dict_2, orient='index', columns=['Feature Importance Model2']))

        # some statistics out of the results dataframe
        all_results['Importance Change'] = all_results['Feature Importance Model2'] - all_results['Feature Importance Model1']
        all_results['% Importance Change'] = all_results['Importance Change'] / all_results['Feature Importance Model1']
        all_results['Rank Model1'] = all_results['Feature Importance Model1'].rank(method='min', ascending=False)
        all_results['Rank Model2'] = all_results['Feature Importance Model2'].rank(method='min', ascending=False)
        all_results['Rank Change'] = all_results['Rank Model1'] - all_results['Rank Model2']
        all_results = all_results.sort_values('Rank Model1')

        self.all_results_ = all_results

        ###########################
        # the below part outputs the features of max. increase/decrease in rank and imp., according to the cutoff point
        assert (cutoff is None) | (0 <= cutoff <= 1), "The values allowed for `output` is `None` or a float [0, 1]."

        # get index of feature at which the cumulative %importances exceeds the cutoff point specified by the user
        counter1 = 0
        init1 = 0
        for value in self.all_results_[['Feature Importance Model1']].values:
            init1 += value
            counter1 += 1
            if init1 > cutoff:
                break

        counter2 = 0
        init2 = 0
        for value in self.all_results_[['Feature Importance Model2']].values:
            init2 += value
            counter2 += 1
            if init2 > cutoff:
                break

        # if both counters are the same, then select a subset of the self.results_ accordingly; if not, go for the
        # counter that demands choosing a larger sub-table, i.e. the larger counter
        if counter1 == counter2:
            results_sub = self.all_results_.iloc[:counter1, :]
        else:
            results_sub = self.all_results_.iloc[:max(counter1, counter2), :]

        # get corresponding features
        max_rank_increase_features = results_sub[results_sub['Rank Change'] == results_sub['Rank Change'].max()].index.tolist()
        min_rank_increase_features = results_sub[results_sub['Rank Change'] == results_sub['Rank Change'].min()].index.tolist()

        # make dict for final results
        analysis_dict = {
            'Results': [['Max Rank Increase'] * len(max_rank_increase_features) +
                        ['Max Rank Decrease'] * len(min_rank_increase_features)][0],
            'Rank in Model1': [results_sub.loc[i, 'Rank Model1'] for i in
                      [i for i in max_rank_increase_features + min_rank_increase_features]],
            'Rank Change': [results_sub.loc[i, 'Rank Change'] for i in
                            [i for i in max_rank_increase_features + min_rank_increase_features]],
            'Importance in Model1': [results_sub.loc[i, 'Feature Importance Model1'] for i in
                                     [i for i in max_rank_increase_features + min_rank_increase_features]],
            'Importance Change': [results_sub.loc[i, 'Importance Change'] for i in
                                  [i for i in max_rank_increase_features + min_rank_increase_features]]
        }

        # turn final results dict into dataframe
        results_ = pd.DataFrame(analysis_dict, index=[max_rank_increase_features + min_rank_increase_features])
        self.results_ = results_

        return self.results_


    def rule_changes(self):

        # asserts
        assert ((type(self.model1) == sklearn.ensemble._forest.RandomForestClassifier) |
                type(self.model2) == sklearn.ensemble._forest.RandomForestClassifier), \
            "Decision rule changes cannot be extracted from RandomForestClassifiers."







