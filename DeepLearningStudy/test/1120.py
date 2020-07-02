import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-5, 5, 100)
print(x)
y = np.sin(x)

plt.plot(x, y)
plt.show()
