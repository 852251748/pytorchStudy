from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

x = np.array([1, 3, 5, 6, 9, 12, 15, 13, 19, 21, 22, 26], dtype=np.float64).reshape(-1, 1)

y = (3 * x + 2).ravel()

y[::2] += 15 * np.random.rand(y.shape[0] // 2)

mode1 = linear_model.LinearRegression()
mode1.fit(x, y)
# CART用于回归时使用的损失函数时MSE均方差，而用于分类时使用的时gini指数或者熵
mode2 = tree.DecisionTreeRegressor(criterion="mse", max_depth=3)
mode2.fit(x, y)
mode3 = tree.DecisionTreeRegressor(max_depth=1)
mode3.fit(x, y)

test_x = np.linspace(0, 26, 12).reshape(-1, 1)
print(test_x)

pred_y1 = mode1.predict(test_x)
pred_y2 = mode2.predict(test_x)
pred_y3 = mode3.predict(test_x)

plt.scatter(x, y, color="blue")
plt.plot(test_x, pred_y1, color="red")
plt.plot(test_x, pred_y2, color="green")
plt.plot(test_x, pred_y3, color="black")
plt.show()
