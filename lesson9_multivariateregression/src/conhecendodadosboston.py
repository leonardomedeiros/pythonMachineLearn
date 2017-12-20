from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
print(boston.data.shape)
print(boston.feature_names)