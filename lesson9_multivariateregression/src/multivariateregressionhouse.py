from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
boston = load_boston()

x = np.array(boston.data)
y = np.array(boston.target)

p4 = np.poly1d(np.polyfit(x, y, 4))
print(boston.target[0])
print(p4([boston.data[0]]))