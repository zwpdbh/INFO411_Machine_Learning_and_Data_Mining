import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from MyLib import *



# import some data to play with
# rng = np.random.RandomState(0)
# n_samples_1 = 1000
# n_samples_2 = 100
# X = np.r_[1.5 * rng.randn(n_samples_1, 2),
#           0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
# y = [0] * (n_samples_1) + [1] * (n_samples_2)
#
# MyLib.compare_weighted_SVM(X, y, C=5.0, class_weight={1: 10})
# plt.show()

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

MyLib.svm_novelty_detection(xx, yy, X_train, X_test, X_outliers, gamma=0.1, plot_graph=False)
plt.show()