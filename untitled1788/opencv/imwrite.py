import cv2 as cv
import numpy as np

img = np.empty((200, 200, 3), np.uint8)
print(img.shape)

img[..., 0] = 255
img[..., 1] = 0
img[..., 2] = 0
img = img[..., ::-1]
cv.imwrite("saveimg.jpg", img)  # 不支持保存没有后缀的图片文件
