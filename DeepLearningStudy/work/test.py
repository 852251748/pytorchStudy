import matplotlib.pyplot as plt
import numpy as np

# def decet(feature, targets, epoch, save_path):
#     color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen", "orange", "blue", "teal"]
#     cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     plt.ion()
#     plt.clf()
#     for j in cls:
#         mask = [targets == j]
#         feature_ = feature[mask].numpy()
#         x = feature_[:, 1]
#         y = feature_[:, 0]
#         label = cls
#         plt.plot(x, y, ".", color=color[j])
#         plt.legend(label, loc="upper right")  # 如果写在plot上面，则标签内容不能显示完整
#         plt.title("epoch={}".format(str(epoch)))
#
#     plt.savefig('{}/{}.jpg'.format(save_path, epoch + 1))
#     plt.draw()
#     plt.pause(0.001)

a = np.array([1, 2, 3, 4])
y = 3 * a + 1

label = "ca"

plt.scatter(a, y, color="red")
plt.legend(label, loc="upper right")
plt.title("epoch={}".format(str(1)))
plt.show()
