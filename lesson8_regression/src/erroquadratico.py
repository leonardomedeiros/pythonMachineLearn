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
erroquadratico = r_value ** 2
print("Erro Quadratico%d", erroquadratico)
