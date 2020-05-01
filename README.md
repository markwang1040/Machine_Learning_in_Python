# Machine Learning in Python
pipelines for faster workflow and cleaner presentation

### 1. process_data: pipeline for processing data

- inputs: list of categorical and continuous variables, and list of variables to be left alone; X <br>
- output: a dataframe X with categorical variables one-hot encoded with drop_first, and continous variables standardized <br>
    
### 2. good_fit: pipeline for hyper parameter tuning

- inputs: objective: {'clf', 'reg'}, base model, X_train, X_test, y_train, y_test, random_state, cross validation folds <br>
- outputs: <br>
        1. a log of parameters with which to tune the models, as well as training and testing error metrics, all cross     -validated with k folds as specified by the user; returned after calling function and stored as attribute <br>
        2. a list of fitted estimators in accordance to the sequence of combinations of parameters; stored as attribute <br>

- the important thing here is that the results are returned as a log as opposed to a single outcome as in sklearn, and that the fitted estimators are not lost but stored, for use through indexing anytime <br>
    
### 3. compare: class and methods for comparing forests/trees

- inputs: 2 fitted models of the same kind:{'RandomForestClassifier', 'ExtraTreeClassifier', 'DecisionTreeClassifier'} <br>
- outputs: a table of feature importances changes and related statistics, and a smaller table of features with the most drastic increase/decrease in feature importances and related statistics <br>

- this is for comparison of two models with the same X but different ys, e.g. in marketing analytics where the X is user profile and activity data, but the ys being different metrics e.g. 2nd day retention binary outcome and 14th day retention binary outcome <br>
      
more updates to follow
    
