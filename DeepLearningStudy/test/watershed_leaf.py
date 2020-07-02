# 分水岭分割法
# cv2.watershed 方法需要传入原图和marsk，marsk需要将不确定区域（包含边界的区域）的像素置为0，其余确定区域置为非零的正整数（使用connectedComponents）
# connectedComponents函数会将确定的区域置为非0的正整数，不确定的区域置为0。（确定的区域就是传入的二值图的白色部分（像素值为255））
#
#
import cv2 as cv
import numpy as np

img = cv.imread("125.png")

# 将图片转换成灰度图
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# 将图片转换成二值图
ret, thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY_INV)

# 执行闭操作进行补洞
kernel = np.ones((3, 3), np.uint8)
imgmo = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# 执行开操作去掉坐标轴
kernel_Op = np.ones((9, 9), np.uint8)
imgmoOp = cv.morphologyEx(imgmo, cv.MORPH_OPEN, kernel_Op)

# 膨胀操作，扩大不确定区域（也就是包含边界的区域），确保边界包含在不确定区域
sure_bg = cv.dilate(imgmo, kernel, iterations=3)

# 腐蚀操作，缩小确定区域，扩大不确定区域，确保边界包含在不确定区域
kernel_fg = np.ones((11, 11), np.uint8)
sure_fg = cv.erode(imgmo, kernel_fg)

# 相减，获得不确定区域
unknow = cv.subtract(sure_bg, sure_fg)

# 获得marsk将确定区域（白色区域）置为非0正整数
_, markers = cv.connectedComponents(sure_fg)
# 整体加1使的背景不要为0，背景像素值为1
markers += 1
# 将不确定区域置为0。
# 就会将不确定区域（包含边界）像素值：0，背景像素值：1，确定区域：像素值非0正整数，都区分开
markers[unknow == 255] = 0

# 传入原图片，与marsk，位置区域的像素值要为0
markers = cv.watershed(img, markers)

# 输出marsk值为-1的地方为边界
img[markers == -1] = [0, 0, 255]

cv.imshow("img", img)
cv.waitKey(0)
exit()
