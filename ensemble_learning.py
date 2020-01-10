import pandas as pd
import numpy as np
import sklearn as sk

# ==================
# Instantiate list of models to be used

SEED = 1

# Instantiate lr
lr = sk.LogisticRegression(random_state=SEED)

# Instantiate knn
knn = sk.KNeighborsClassifier(n_neighbors=27)

# Instantiate dt
dt = sk.tree.DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# And so on

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# ==================
# Iterate over the pre-defined list of classifiers

for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = sk.metrics.accuracy_score(y_pred, y_test)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# ==================
# Construct and run voting classifier

# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_pred, y_test)
print('Voting Classifier: {:.3f}'.format(accuracy))
