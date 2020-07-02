# 核岭回归
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
x = 5 * rng.rand(100, 1)
y = np.sin(x)
print(x.shape, y.shape)

# kr = KernelRidge(kernel='linear', gamma=0.4)
kr = GridSearchCV(KernelRidge(), param_grid={"kernel": ["rbf", "laplacian", "sigmoid", "polynomial"],
                                             "alpha": [1e0, 0.1, 1e-2, 1e-3],
                                             "gamma": np.logspace(-2, 2, 5)})
kr.fit(x, y)
print(kr.best_score_, kr.best_params_)
test_x = np.linspace(0, 5, 100).reshape(-1, 1)
# print(test_x.shape)
# exit()
pred_y = kr.predict(test_x)

plt.scatter(x, y)
plt.plot(test_x, pred_y, color="red", linewidth=3.0)
plt.show()
