import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.tree import H2OTree



class Woodsmith:

    def __init__(self, model1, model2):
        """
        Takes in two models for comparison.

        The two forests should have been trained on the same features. If the inputs are objects from the sklearn package,
        then the names of the features before and after the one-hot encoding of the features should have been already
        stored in the instances of the object, as `model.feature_names_before_` and `model.feature_names_after_` respectively.

        :param
            model1: fitted instance of forest object, of type [sklearn.ensemble._forest.RandomForestClassifier,
            h2o.estimators.random_forest.H2ORandomForestEstimator, h2o.estimators.generic.H2OGenericEstimator]
        :param
            model2: fitted instance of  forest object, of type [sklearn.ensemble._forest.RandomForestClassifier,
            h2o.estimators.random_forest.H2ORandomForestEstimator, h2o.estimators.generic.H2OGenericEstimator]

        Attributes:
            model1_, model2_: the two arguments stored
        """
        assert type(model1) == type(model2), "The two models have to be of the same type for comparison."
        assert type(model1) in [sklearn.ensemble._forest.RandomForestClassifier,
                                sklearn.tree._classes.DecisionTreeClassifier, sklearn.tree._classes.ExtraTreeClassifier,
                                h2o.estimators.random_forest.H2ORandomForestEstimator,
                                h2o.estimators.generic.H2OGenericEstimator], "Model1 is of an unspportued type."
        assert type(model2) in [sklearn.ensemble._forest.RandomForestClassifier,
                                sklearn.tree._classes.DecisionTreeClassifier, sklearn.tree._classes.ExtraTreeClassifier,
                                h2o.estimators.random_forest.H2ORandomForestEstimator,
                                h2o.estimators.generic.H2OGenericEstimator], "Model1 is of an unspportued type."

        if type(model1) in [sklearn.ensemble._forest.RandomForestClassifier,
                                sklearn.tree._classes.DecisionTreeClassifier, sklearn.tree._classes.ExtraTreeClassifier]:
            assert ('feature_names_before_' and 'feature_names_after_') in dir(model1), \
                "`feature_names_before` and `feature_names_after` not found in model1 object."
            assert ('feature_names_before_' and 'feature_names_after_') in dir(model2), \
                "`feature_names_before_` and `feature_names_after` not found in model2 object."

            assert model1.feature_names_before_ == model2.feature_names_before_, \
                "The two models have different feature names before data is processed."
            assert model1.feature_names_after_ == model2.feature_names_after_, \
                "The two models have different feature names after data is processed."

        self._model1 = model1
        self._model2 = model2


    """
    The below code prevents the user from resetting the models associated with the current instance of the object.
    """
    @property
    def model1(self):
        return self._model1

    @model1.setter
    def model1(self, new_model):
        raise ModelResetException("The models cannot be changed; "
                                  "instantiate another Woodsmith object if you are trying to compare other models. ")

    @property
    def model2(self):
        return self._model2

    @model2.setter
    def model2(self, new_model):
        raise ModelResetException("The models cannot be changed; "
                                  "instantiate another Woodsmith object if you are trying to compare other models. ")


    def longestCommonPrefix(self, m):
        """
        Slick, Pythonic LCP function for use in later functions.

        :param
            m (list): list of strings from which to search for the LCP.
        :return:
            LCP (string): the longest common prefix of all strings in the list.
        """
        assert type(m) == list, "The input argument needs to be a list of strings."
        for i in m:
            assert type(i) == str, "The input argument needs to be a list of strings."

        # making use of the fact that the list of strings will be sorted and items retrieved min/max by alphabetical order
        s1 = min(m)
        s2 = max(m)

        for i, c in enumerate(s1):
            if c != s2[i]:
                # stop until the split index is reached
                LCP = s1[:i]
                return LCP
        LCP = s1
        return LCP


    def importance_changes(self):
        """
        Takes two fitted forests and outputs a dataframe, detailing the changes of the importance of each feature used
        in training the two forests.

        The two forests should have been trained on the same features. If the inputs are objects from the sklearn package,
        then the names of the features before and after the one-hot encoding of the features should have been already
        stored in the instances of the object, as `model.feature_names_before_` and `model.feature_names_after_` respectively.

        :return:
            all_results_ (pandas DataFrame): with feature names before one-hot as the index, and columns being their importances in
            each model, the absolute change in importances across the two models, the percentage change of importances
            across two models, the importance ranks in each model (the most important feature is ranked 1), and the change
            in rank across the two modes

        Attributes:
            all_results_
            max_rank_increase_features_: list of features with the maximum increase in importance ranks
            min_rank_increase_features_: list of features with the maximum decrease in importance ranks
        """

        # sklearn
        if type(self._model1) in [sklearn.ensemble._forest.RandomForestClassifier,
                                sklearn.tree._classes.DecisionTreeClassifier, sklearn.tree._classes.ExtraTreeClassifier]:

            self._model_type = 'sklearn'

            # extract feature importances
            f_imp1 = self._model1.feature_importances_
            f_imp2 = self._model2.feature_importances_

            # construct feature importances dicts, with feature names being those after one-hot encoding
            f_imp_scattered_dict_1 = {key:value for key, value in zip(self._model1.feature_names_after_, f_imp1)}
            f_imp_scattered_dict_2 = {key:value for key, value in zip(self._model2.feature_names_after_, f_imp2)}

            # construct feature importances dicts with the original feature names; values initiated at 0
            f_imp_collated_dict_1 = {key:0 for key in self._model1.feature_names_before_}
            f_imp_collated_dict_2 = {key:0 for key in self._model2.feature_names_before_}

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
            # `display` works for Jupyter Notebooks
            try:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    display(self.all_results_)
            except:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(self.all_results_)
            return self.all_results_

        # h2o
        else:

            self._model_type = 'h2o'

            f_imp_1 = self._model1._model_json['output']['variable_importances'].as_data_frame()
            f_imp_2 = self._model2._model_json['output']['variable_importances'].as_data_frame()

            all_results = f_imp_1.join(f_imp_2.set_index('variable'), on='variable', lsuffix='_model1', rsuffix='_model2')

            all_results['Feature Importance Model1'] = all_results['percentage_model1']
            all_results['Feature Importance Model2'] = all_results['percentage_model2']
            all_results['Importance Change'] = all_results['Feature Importance Model2'] - all_results[
                'Feature Importance Model1']
            all_results['% Importance Change'] = all_results['Importance Change'] / all_results[
                'Feature Importance Model1']
            all_results['Rank Model1'] = all_results['Feature Importance Model1'].rank(method='min', ascending=False)
            all_results['Rank Model2'] = all_results['Feature Importance Model2'].rank(method='min', ascending=False)
            all_results['Rank Change'] = all_results['Rank Model1'] - all_results['Rank Model2']
            all_results = all_results.sort_values('Rank Model1')
            all_results = all_results.drop(
                ['relative_importance_model1', 'relative_importance_model2', 'scaled_importance_model1',
                 'scaled_importance_model2', 'percentage_model1', 'percentage_model2'], axis=1).set_index('variable')
            all_results.index.name = None

            self.all_results_ = all_results

            # `display` works for Jupyter Notebooks
            try:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    display(self.all_results_)
            except:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(self.all_results_)

            return self.all_results_


    def specific_importance_changes(self, cutoff=1, features=[]):
        """
        Returns specific importance changes, which is a subset of `self.all_results_`, based on `cutoff`.

        :param
            cutoff (float [0, 1]): if `features` is not empty, features' importances will be summed, starting from the
            most important feature, and the process will stop when the cumsum of the importances equate to or exceed
            `cutoff`.
        :return:
            specific_results_ (pandas DataFrame): index is the features with the maximum increase/decrease in importance
            rank across the two models, selected according to `cutoff`, and columns their ranks in the first model,
            the change in rank, their importances in the first model and the absolute change in importance across
            the two models.
        """

        # check compliance of `cutoff`
        assert (0 <= cutoff <= 1), "The values allowed for `output` is a float within the range [0, 1]."

        # get index of feature at which the cumulative %importances exceeds the cutoff point specified by the user
        counter1 = 0
        init1 = 0
        for value in self.all_results_[['Feature Importance Model1']].values:
            init1 += value
            counter1 += 1
            if init1 >= cutoff:
                break

        counter2 = 0
        init2 = 0
        for value in self.all_results_[['Feature Importance Model2']].values:
            init2 += value
            counter2 += 1
            if init2 >= cutoff:
                break

        # if both counters are the same, then select a subset of the self.results_ accordingly; if not, go for the
        # counter that demands choosing a larger sub-table, i.e. the larger counter
        if counter1 == counter2:
            results_sub = self.all_results_.iloc[:counter1, :]
        else:
            results_sub = self.all_results_.iloc[:max(counter1, counter2), :]

        # get corresponding features
        max_rank_increase_features = results_sub[
            results_sub['Rank Change'] == results_sub['Rank Change'].max()].index.tolist()
        min_rank_increase_features = results_sub[
            results_sub['Rank Change'] == results_sub['Rank Change'].min()].index.tolist()

        # store max_ and min_rank_increase_features for use in rule_changes()
        self.max_rank_increase_features_ = max_rank_increase_features
        self.min_rank_increase_features_ = min_rank_increase_features

        # make dict for final results
        analysis_dict = {
            'Result': [['Max Rank Increase'] * len(max_rank_increase_features) +
                        ['Max Rank Decrease'] * len(min_rank_increase_features)][0],
            'Rank Model1': [results_sub.loc[i, 'Rank Model1'] for i in
                               [i for i in max_rank_increase_features + min_rank_increase_features]],
            'Rank Change': [results_sub.loc[i, 'Rank Change'] for i in
                            [i for i in max_rank_increase_features + min_rank_increase_features]],
            'Feature Importance Model1': [results_sub.loc[i, 'Feature Importance Model1'] for i in
                                     [i for i in max_rank_increase_features + min_rank_increase_features]],
            'Importance Change': [results_sub.loc[i, 'Importance Change'] for i in
                                  [i for i in max_rank_increase_features + min_rank_increase_features]]
        }

        # turn final results dict into dataframe
        specific_results_ = pd.DataFrame(analysis_dict, index=[max_rank_increase_features + min_rank_increase_features])
        self.specific_results_ = specific_results_

        # `display` works for Jupyter Notebook
        try:
            display(self.specific_results_)
        except:
            print(self.specific_results_)

        return self.specific_results_


    def importances_changes_subset(self, features=[]):
        """
        Returns a subset of the results obtained by running self.importance_changes().

        :param
            features (list): a list of features, used to select rows from self.all_results_ to return;
        :return:
            all_results_subset_ (pandas DataFrame): results, including the features' ranks in model1, their rank changes,
            their importances in model1 and the importance changes.
        """
        # check if all features in `features` are related to the model of concern
        for feature in features:
            if self._model_type == 'sklearn':
                assert feature in self._model1.feature_names_before_, "Feature {} is not in the model.".format(feature)
            else:
                assert feature in self.all_results_.index.tolist(), "Feature {} is not in the model.".format(feature)

        all_results_subset_ = self.all_results_.loc[features, ['Rank Model1', 'Rank Change',
                                                             'Feature Importance Model1', 'Importance Change']]
        self.all_results_subset_ = all_results_subset_
        return self.all_results_subset_

























    def rule_changes(self, tree1, tree2):
        """

        Outputs the decision rule changes across the two input decision trees, of features in self.max_rank_increase_features_
        and self.min_rank_increase_features_.

        Deprecated after meeting on 8 May 2020.

        :param
            tree1: a fitted decision tree of type [sklearn.tree._classes.DecisionTreeClassifier,
            h2o.estimators.random_forest.H2ORandomForestEstimator, h2o.estimators.generic.H2OGenericEstimator,
            h2o.estimators.gbm.H2OGradientBoostingEstimator]
        :param
            tree2: a fitted decision tree of type [sklearn.tree._classes.DecisionTreeClassifier,
            h2o.estimators.random_forest.H2ORandomForestEstimator, h2o.estimators.generic.H2OGenericEstimator,
            h2o.estimators.gbm.H2OGradientBoostingEstimator]
        :return:
            rule_changes_max_, rule_changes_min_: dictionaries of feature:rule_changes; empty dict if the feature is categorical
            and there is no rule change
        """

        assert type(tree1) in [sklearn.tree._classes.DecisionTreeClassifier, h2o.estimators.random_forest.H2ORandomForestEstimator,
                                  h2o.estimators.generic.H2OGenericEstimator, h2o.estimators.gbm.H2OGradientBoostingEstimator], \
            "tree1 is of an unsupported type."
        assert type(tree2) in [sklearn.tree._classes.DecisionTreeClassifier, h2o.estimators.random_forest.H2ORandomForestEstimator,
                               h2o.estimators.generic.H2OGenericEstimator, h2o.estimators.gbm.H2OGradientBoostingEstimator], \
            "tree2 is of an unsupported type."
        assert type(tree1) == type(tree2), "The two trees are not of the same type."

        self.tree1_ = H2OTree(model=tree1, tree_number=0)
        self.tree2_ = H2OTree(model=tree2, tree_number=0)

        # h2o
        if type(self.tree1_) in [h2o.estimators.random_forest.H2ORandomForestEstimator, h2o.estimators.generic.H2OGenericEstimator,
                                 h2o.estimators.gbm.H2OGradientBoostingEstimator, h2o.tree.tree.H2OTree]:
            # max
            rules_max_1 = []
            rules_max_2 = []
            rules_min_1 = []
            rules_min_2 = []

            for feature in self.max_rank_increase_features_:
                for i in range(len(self.tree1_.features)):
                    if self.tree1_.features[i] == feature:
                        rules_max_1.append(self.tree1_.descriptions[i])
                        break

                for i in range(len(self.tree2_.features)):
                    if self.tree2_.features[i] == feature:
                        rules_max_2.append(self.tree2_.descriptions[i])
                        break

            for feature in self.min_rank_increase_features_:
                for i in range(len(self.tree1_.features)):
                    if self.tree1_.features[i] == feature:
                        rules_min_1.append(self.tree1_.descriptions[i])
                        break

                for i in range(len(self.tree2_.features)):
                    if self.tree2_.features[i] == feature:
                        rules_min_2.append(self.tree2_.descriptions[i])
                        break

            self.rules_max_1_ = rules_max_1
            self.rules_max_2_ = rules_max_2
            self.rules_min_1_ = rules_min_1
            self.rules_min_2_ = rules_min_2

            rules_diff_max = []
            rules_diff_min = []

            for i in range(len(rules_max_1)):
                if 'categorical' in rules_max_1[i].split('node')[2].strip().split(' '):
                    rule1 = rules_max_1[i].split('node')[2].split(':')[1].split('.')[0].strip().split(',')
                    rule2 = rules_max_2[i].split('node')[2].split(':')[1].split('.')[0].strip().split(',')
                    rule_diff = list(set(rule1) - set(rule2))
                    rules_diff_max.append(', '.join(rule_diff))
                else:
                    rule1 = float(rules_max_1[i].split('node')[1].split(' ')[6])
                    rule2 = float(rules_max_2[i].split('node')[1].split(' ')[6])
                    rule_diff = str(rule1) + ' -> ' + str(rule2)
                    rules_diff_max.append(rule_diff)

            for i in range(len(rules_min_1)):
                if 'categorical' in rules_min_1[i].split('node')[2].strip().split(' '):
                    rule1 = rules_min_1[i].split('node')[2].split(':')[1].split('.')[0].strip().split(',')
                    rule2 = rules_min_2[i].split('node')[2].split(':')[1].split('.')[0].strip().split(',')
                    rule_diff = list(set(rule1) - set(rule2))
                    rules_diff_min.append(', '.join(rule_diff))
                else:
                    rule1 = float(rules_min_1[i].split('node')[1].split(' ')[6])
                    rule2 = float(rules_min_2[i].split('node')[1].split(' ')[6])
                    rule_diff = str(rule1) + ' -> ' + str(rule2)
                    rules_diff_min.append(rule_diff)

            # if len(rules_diff_max) == 0:
            #     rule_change_dict_max = {i:'No change' for i in self.max_rank_increase_features_}
            # else:
            rule_change_dict_max = {i:j for i, j in zip(self.max_rank_increase_features_, rules_diff_max)}

            # if len(rules_diff_min) == 0:
            #     rule_change_dict_min = {i: 'No change' for i in self.min_rank_increase_features_}
            # else:
            rule_change_dict_min = {i:j for i, j in zip(self.min_rank_increase_features_, rules_diff_min)}

            self.rule_changes_max_ = rule_change_dict_max
            self.rule_changes_min_ = rule_change_dict_min
            self.rules_diff_max_ = rules_diff_max
            self.rules_diff_min_ = rules_diff_min

            try:
                display(rule_change_dict_max)
                display(rule_change_dict_min)
            except:
                print(rule_change_dict_max)
                print(rule_change_dict_min)

            return self.rule_changes_max_, self.rule_changes_min_

        #sklearn
        else:
            pass
