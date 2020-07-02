import cv2
import numpy as np

src = cv2.imread(r"1.jpg")
# kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], np.float32)  # 定义一个核
# dst = cv2.filter2D(src, -1, kernel=kernel)

# dst = cv2.blur(src,(3,3))
dst = cv2.GaussianBlur(src, (3, 3), 1)

cv2.imshow("src show", src)
cv2.imshow("dst show", dst)
cv2.waitKey(0)
