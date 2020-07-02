# 聚类
from sklearn.datasets import samples_generator
from sklearn import metrics, cluster
import matplotlib.pyplot as plt

# x, y_lable = samples_generator.make_blobs(n_samples=200, centers=2, cluster_std=0.60, random_state=0)
# x, y_lable = samples_generator.make_moons(200, noise=0.05, random_state=0)
x, y_lable = samples_generator.make_circles(200, noise=0.05, random_state=0, factor=0.4)

# Kmeans/Kmeans++聚类
# model = cluster.KMeans(2)
# 密度聚类
# model = cluster.DBSCAN(eps=0.3, min_samples=5)
# meanShift
# model = cluster.MeanShift()
# AP聚类
# model = cluster.AffinityPropagation()
# 层次聚类.
model = cluster.AgglomerativeClustering()

y_pred = model.fit_predict(x)
# model.fit(x)
# y_pred = model.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()
