import pandas as pd
import numpy as np
#the csv file is located in a previous subdirectory called tables
train_dataset = pd.read_csv('../tables/dadosprecocivic.csv',';')
print(train_dataset['kilometragem-x'])
X = train_dataset['kilometragem-x']
Y = train_dataset['preco-y']
mediaX = np.mean(X)
mediaY = np.mean(Y)
stdX = np.std(train_dataset['kilometragem-x'], ddof=1)
stdY = np.std(train_dataset['preco-y'], ddof=1)
#ddof : int, optional
    #Means Delta Degrees of Freedom. The divisor used in calculations is
    #N - ddof, where N represents the number of elements. By default ddof is zero.
#Because this parameter we get the diference between Excel and Numpy result.
print("Media X %d", mediaX)
print("Media Y %d",mediaY)
print("Desvio Padrao X %d",stdX)
print("Desvio Padrao Y %d",stdY)
deviationX = X - mediaX
deviationY = Y - mediaY
sumCovariance = np.sum(deviationX*deviationY)
print("Soma da Covariancia %d", sumCovariance)
#Number of instances
N = train_dataset.shape[0]
print(N)
pearsonCorrelation = sumCovariance/((N-1)*stdX*stdY)
print("Correlacao %.4f", pearsonCorrelation)

beta = pearsonCorrelation * (stdY/stdX)
alpha = mediaY-(beta*mediaX)
print("Beta %d", beta)
print("Alpha %d", alpha)
