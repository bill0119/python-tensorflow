# lab5_sklearn_classification

# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Information:
#
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica


# conda info --envs
# activate tensorflow_08_04
# pip install matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# load dataset iris
iris = datasets.load_iris()
print(list(iris.keys()))
x = iris['data'][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)

print("data x=", x[:110])
print("target y=", y[:110])

log_regression = LogisticRegression()
log_regression.fit(x, y)

# from 0 t0 3 have 1000 point
x_seq = np.linspace(0, 3, 1000)
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print("x_seq", x_seq[:50])
print("x_new", x_new[:50])
y_prob = log_regression.predict_proba(x_new)

plt.plot(x, y, 'g*')
plt.plot(x_new, y_prob[:,1], 'b-', label="Iris-virginica")
plt.plot(x_new, y_prob[:,0], 'r--', label="Iris, not virginica")
plt.xlabel("Petal width", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.show()
