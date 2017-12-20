from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

boston_dataset = load_boston()
datafeatures = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
targetvalues = boston_dataset.target
#print(boston_dataset.data.shape)
#print(datafeatures.describe())
#print(targetvalues)
print(boston_dataset.feature_names)
np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)
#print(boston_dataset.data)
#first 10 elements
print(boston_dataset.data[:10])
print(targetvalues[:10])



# #datasplit
X_train, X_test, y_train, y_test = train_test_split(datafeatures, targetvalues, random_state=0)

#trainning the regressor
#regressor = KNeighborsRegressor(n_neighbors=3) #nao deu certo 51.23

regressor = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
 metric_params=None, n_jobs=1, n_neighbors=3, p=2,
 weights='uniform')
regressor.fit(X_train, y_train)


predictedvalues = regressor.predict(X_test)
score = regressor.score(X_test, y_test)
print(predictedvalues)
print(score)
# accuracy = regressor.score(X_test, y_test)
# print(accuracy)

#evaluate the regression
# Now we can constuct a vector of errors
# err = abs(predictedvalues-y_test)

# # Let's see the error on the first 10 predictions
# print("Error values")
# print(err)

# # Dot product of error vector with itself gives us the sum of squared errors
# total_error = np.dot(err,err)
# # Compute RMSE
# rmse_train = np.sqrt(total_error/len(predictedvalues))
# print("rmse_train")
# print rmse_train


