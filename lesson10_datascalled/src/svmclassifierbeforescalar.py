from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
  random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Teste de acuracia: {:.2f}".format(svm.score(X_test, y_test)))