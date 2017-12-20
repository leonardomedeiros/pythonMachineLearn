from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

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


#Create a Linear Regression
#http://facweb.cs.depaul.edu/mobasher/classes/CSC478/Notes/IPython%20Notebook%20-%20Regression.html
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(boston_dataset.data, targetvalues)

predictedvalues = linreg.predict(boston_dataset.data[:10])
print("target values")
print(targetvalues[:10])
print("predictedvalues values")
print(predictedvalues)

#evaluate the regression
# Now we can constuct a vector of errors
err = abs(predictedvalues-targetvalues[:10])

# Let's see the error on the first 10 predictions
print("Error values")
print err[:10]

# Dot product of error vector with itself gives us the sum of squared errors
total_error = np.dot(err,err)
# Compute RMSE
rmse_train = np.sqrt(total_error/len(predictedvalues))
print("rmse_train")
print rmse_train

print("Regression Coeficients:")
print(linreg.coef_)

score = linreg.score(boston_dataset.data[:10], predictedvalues)
print("score: %d"%score)

#PLOT the Prediction Values
plot.plot(predictedvalues, targetvalues[:10],'ro')
plot.plot([0,50],[0,50], 'g-')
plot.xlabel('predicted')
plot.ylabel('real')
plot.show()
print("matplotlib")


