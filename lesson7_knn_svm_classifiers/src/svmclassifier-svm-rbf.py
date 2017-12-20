from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm 
import numpy as np
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
 random_state=0)
C = 1.0  # SVM regularization parameter
# # SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
y_pred = rbf_svc.predict(X_test)
print(y_pred)
print(y_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)

