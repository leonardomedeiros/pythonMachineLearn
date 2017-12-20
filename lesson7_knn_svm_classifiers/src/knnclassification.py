import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris_datasset = datasets.load_iris()
iris_X = iris_datasset.data
iris_y = iris_datasset.target
print("Iris Label Values")
print(np.unique(iris_y))
# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
print("indices")
print(indices)
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]
print("test labels")
print(iris_y_test)
# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
predictedclasses = knn.predict(iris_X_test)
print("Predicted Instances")
print(predictedclasses)
print("Iris Label Values")
print(iris_y_test)
classifier_accuracy = np.mean(predictedclasses == iris_y_test)
print("Classifier Accuracy")
print(classifier_accuracy)