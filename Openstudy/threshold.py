import cv2

img = cv2.imread("src1.jpg")
img = cv2.resize(img, None, img, 0.5, 0.5)
# 先转换成会对图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 加cv2.THRESH_OTSU意思是,使用OTSU算法计算一个合适的阀值，然后根据这个阀值进行二值化,第二个参数就填成0，否则的话第二个入参就填成自定义的阀值
retval1, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 小于127的保持变成白色，大于127的变成黑色
# retval, dst = cv2.threshold(img, 127, 255,  cv2.THRESH_BINARY)
# 与THRESH_BINARY原理相反
# retval, dst = cv2.threshold(img, 127, 255,  cv2.THRESH_BINARY_INV)
# 小于127的保持原色，大于127的变成黑色
# retval, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# 与THRESH_TOZERO相反 ，小于127的变成黑色，大于127的保持原色
# retval, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV )
# 像素小于127的变成白色大于127的保持原色
# retval, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

cv2.imshow("src", img)
cv2.imshow("dst", dst)

cv2.waitKey(0)
