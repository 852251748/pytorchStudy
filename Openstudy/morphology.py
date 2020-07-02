import cv2

img = cv2.imread("4.jpg")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 开操作
# dst = cv2.dilate(img, kernel)
# 闭操作
# dst = cv2.erode(img, kernel)
# 开操作 去噪点
# dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 闭操作 补漏洞的
# dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# 梯度操作 找轮廓的
# dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# 顶帽操作 原图像减去开操作 找噪点的
# dst = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# 黑帽操作  找漏洞的 原图像减去闭操作
dst = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("img", img)
cv2.imshow("dst", dst)
cv2.waitKey(0)
