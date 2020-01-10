import sklearn
import pandas as pd
import numpy as np

# classification tree in sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# instantiate decision tree

dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# fit data

dt.fit(X_train, X_test)

# predict results

y_pred = dt.predict(y_train)

# evaluate test accuracy

accuracy_score(y_test, y_pred)

