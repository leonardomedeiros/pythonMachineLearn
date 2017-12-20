from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
cancer = load_breast_cancer()

#X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
#X_train, X_test, y_train, y_test = train_test_split(cancer
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', label="Test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', label="Test set", s=60)
axes[1].set_title("Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
fig.tight_layout()
plt.savefig('../img/plotdatascaled.png')
plt.show()
