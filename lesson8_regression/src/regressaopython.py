import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#the csv file is located in a previous subdirectory called tables
train_dataset = pd.read_csv('../tables/dadosprecocivic.csv',';')
X = train_dataset['kilometragem-x']
Y = train_dataset['preco-y']
#slope = inclinacao
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
print("(Beta) Inclinacao %d", slope)
print("(Alpha) Intercept %d", intercept)
print("Coeficiente  Correlacao %d", r_value)
#two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero
print("P Value %d", p_value)
#Standard error of the estimated gradient.
print("Erro do Desvio Padrao %d", std_err)
plt.title('Modelo de Regressao Linear do Civic')
plt.xlabel('Kilometragem')
plt.ylabel('Preco')
plt.plot(X, Y, 'o', label = 'original data')
plt.plot(X, intercept + slope*X , 'r', label='fitted line')
plt.legend()
plt.savefig('../img/plotlinearregression.png')
plt.show()
