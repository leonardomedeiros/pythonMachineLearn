# Import necessary packages
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
boston = datasets.load_boston()
yb = boston.target.reshape(-1, 1)
Xb = boston['data'][:,5].reshape(-1, 1)

# Plot data
plt.scatter(Xb,yb)
plt.ylabel('value of house /1000 ($)')
plt.xlabel('number of rooms')

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit( Xb, yb)

# Plot outputs
plt.scatter(Xb, yb,  color='black')
plt.plot(Xb, regr.predict(Xb), color='blue',
         linewidth=3)
plt.show()

#Evaluate Classifier Score
predictedvalues = regr.predict(Xb)
#score = regr.score(Xb, predictedvalues)
#print(predictedvalues)
#print(score)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(yb, predictedvalues))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(yb, predictedvalues))
