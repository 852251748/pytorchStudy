import cv2
import numpy as np

img = np.empty((200, 200, 3), np.uint8)
img[..., 0] = 0
img[..., 1] = 255
img[..., 2] = 0
img = img[..., :: -1]
cv2.imwrite("save_img.jpg", img)
