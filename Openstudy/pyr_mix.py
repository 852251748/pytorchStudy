import cv2 as cv
import numpy as np

imgap = cv.imread("21.jpg")
imgor = cv.imread("22.jpg")

A = imgap.copy()
gpA = [A]

for i in range(6):
    A = cv.pyrDown(A)
    gpA.append(A)

B = imgor.copy()
gpB = [B]
for i in range(6):
    B = cv.pyrDown(B)
    gpB.append(B)

laA = [gpA[6]]
for i in range(6, 0, -1):
    print(i)
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i - 1], GE)
    laA.append(L)

laB = [gpB[6]]
for i in range(6, 0, -1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i - 1], GE)
    laB.append(L)

LS = []
for i, (la, lb) in enumerate(zip(laA, laB)):
    rows, cols, apt = la.shape
    ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 7):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])

print(len(laA))

cv.imshow("test", np.hstack((imgap[:, :cols // 2], imgor[:, cols // 2:])))
cv.imshow("mix",ls_)
cv.waitKey(0)
