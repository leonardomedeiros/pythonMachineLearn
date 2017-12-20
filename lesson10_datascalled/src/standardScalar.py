from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
scaler = StandardScaler()
# calling fit and transform in sequence (using method chaining)
X_train_scaled = scaler.fit(X_train).transform(X_train)
print(X_train)
# transform data
X_train_scaled = scaler.transform(X_train)
print(X_train)
print(X_train_scaled)
print("per-feature minimum before scaling:\n {}", X_train.min())
print("per-feature maximum before scaling:\n {}", X_train.max())
print("per-feature minimum after scaling:\n {}", X_train_scaled.min())
print("per-feature maximum after scaling:\n {}", X_train_scaled.max())