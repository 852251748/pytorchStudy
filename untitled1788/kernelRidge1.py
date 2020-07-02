import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot as plt

rng = np.random.RandomState(0)

x = 6 * rng.rand(100, 1)
y = np.sin(x).ravel()

y[::5] += 3 * (0.5 - rng.rand(x.shape[0] // 5))
# kr = KernelRidge(kernel='rbf', gamma=0.5)
kr = GridSearchCV(KernelRidge(),
                  param_grid={"kernel": ["rbf", "sigmoid", "polynomial", "laplacian"],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
kr.fit(x, y)

print(kr.best_score_, kr.best_params_)
x_test = np.linspace(0, 6, 120).reshape(-1, 1)
y_pred = kr.predict(x_test)

plt.scatter(x, y)
plt.plot(x_test, y_pred, color='red', linewidth=3.0)
plt.show()
