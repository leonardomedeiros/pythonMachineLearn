from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_train)
X_train
print("per-feature minimum before scaling:\n {}", X_train.min())
print("per-feature maximum before scaling:\n {}", X_train.max())


