from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import crosstab
np.random.seed(0)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())
X_train, X_test, Y_train, Y_test = train_test_split(iris['data'], iris['target'],random_state=0)
#Create RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, Y_train)
predictedClass = clf.predict(X_test)
print('PredictedClasses')
print(predictedClass)
print('Actual Classes')
print(Y_test)
#Create Confusion Matrix
print(pd.crosstab(Y_test, predictedClass, rownames=['Actual Species'], colnames=['Predicted Species']))
  