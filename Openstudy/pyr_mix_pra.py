import cv2 as cv
import numpy as np

imgap = cv.imread("21.jpg")
imgor = cv.imread("22.jpg")

A = imgap.copy()
gaA = [A]
for i in range(6):
    A = cv.pyrDown(A)
    gaA.append(A)

B = imgor.copy()
gaB = [B]
for i in range(6):
    B = cv.pyrDown(B)
    gaB.append(B)

laA = [gaA[6]]
for i in range(6, 0, -1):
    la = cv.pyrUp(gaA[i])
    la = cv.subtract(gaA[i - 1], la)
    laA.append(la)

laB = [gaB[6]]
for i in range(6, 0, -1):
    la = cv.pyrUp(gaB[i])
    la = cv.subtract(gaB[i - 1], la)
    laB.append(la)

LS = []
for i, (la, lb) in enumerate(zip(laA, laB)):
    rows, cols, _ = la.shape
    ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 7):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])

cv.imshow("test", ls_)
cv.waitKey(0)
