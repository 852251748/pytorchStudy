import cv2 as cv

img = cv.imread("23.jpg")

# 高斯模糊 去噪点
img = cv.GaussianBlur(img, (3, 3), 2)

# 灰度化
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("imgGray", imgGray)

# 二值化
ret, thresh = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

# 闭操作：将目标区域连成一块 以便后续提取轮廓
kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 13))
imgmorClose = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv.findContours(imgmorClose, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for item in contours:
    rect = cv.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 2):
        # 裁剪区域图片
        print("x = ",x,",y=", y,",wei=", weight,",hei=", height)
        chepai = img[y:y + height, x:x + weight]
        cv.imshow('chepai' + str(x), chepai)

dst = cv.drawContours(img, contours, -1, (255, 0, 0), 2)
cv.imshow("dst", dst)
cv.waitKey(0)
