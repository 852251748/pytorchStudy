import cv2 as cv
import numpy as np

# org = cv.imread("22.jpg")
# app = cv.imread("21.jpg")
#
# Borg = org.copy()
# Bapp = app.copy()
#
# Gorg = [Borg]
#
# # 橘子高斯金子塔
# for i in range(6):
#     Borg = cv.pyrDown(Borg)
#     Gorg.append(Borg)
#
# Gapp = [Bapp]
#
# # 苹果高斯金子塔
# for i in range(6):
#     Bapp = cv.pyrDown(Bapp)
#     Gapp.append(Bapp)
#
# # 橘子拉普拉斯金子塔
# Lorg = [Gorg[5]]
# for i in range(5, 0, -1):
#     img = cv.pyrUp(Gorg[i])
#     img = cv.subtract(Gorg[i - 1], img)
#     Lorg.append(img)
#
# # 苹果拉普拉斯金子塔
# Lapp = [Gapp[5]]
# for i in range(5, 0, -1):
#     img = cv.pyrUp(Gapp[i])
#     img = cv.subtract(Gapp[i - 1], img)
#     Lapp.append(img)
#
# # 对金字塔每层的苹果橘子各切一半进行合并
# LS = []
# for i, (lo, la) in enumerate(zip(Lorg, Lapp)):
#     col = lo.shape[1]
#     img = np.hstack((la[:, 0: col // 2], lo[:, col // 2:]))
#     LS.append(img)
#
# ls_ = []
# # 将各层图片都加起来
# img = LS[0]
# for i in range(1, 6):
#     img = cv.pyrUp(img)
#     img = cv.add(img, LS[i])
#     ls_.append(img)
#
# cv.imshow("img", ls_[4])
# cv.waitKey(0)


A = cv.imread("21.jpg")
O = cv.imread("22.jpg")

# 苹果的高斯金字塔
CopyA = A.copy()
Ga = [CopyA]
for i in range(6):
    CopyA = cv.pyrDown(CopyA)
    Ga.append(CopyA)

# 下采样分辨率会降低，上采样不会损失分辨率
# 橘子的高斯金字塔
CopyO = O.copy()
Go = [CopyO]
for i in range(6):
    CopyO = cv.pyrDown(CopyO)
    Go.append(CopyO)

# 苹果拉普拉斯金字塔
LaA = [Ga[5]]
for i in range(5, 0, -1):
    img = cv.pyrUp(Ga[i])
    img = cv.subtract(Ga[i - 1], img)
    LaA.append(img)

# 橘子拉普拉斯金字塔
LaO = [Go[5]]
for i in range(5, 0, -1):
    img = cv.pyrUp(Go[i])
    img = cv.subtract(Go[i - 1], img)
    LaO.append(img)

Las = []
for i, (a, o) in enumerate(zip(LaA, LaO)):
    row, col, _ = a.shape
    img = np.hstack((a[:, :col // 2], o[:, col // 2:]))
    Las.append(img)
    # cv.imshow(f"xx{i}", img)
    # cv.waitKey(0)

img = Las[0]
ls_ = []
for i in range(1, 6):
    img = cv.pyrUp(img)
    img = cv.add(img, Las[i])
    ls_.append(img)


cv.imshow("img", ls_[4])
cv.waitKey(0)
