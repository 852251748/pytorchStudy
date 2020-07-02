import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("riya.jpg", 0)
print(img.shape)

# 傅里叶变换
f = np.fft.fft2(img)

# 把中心点移动到中间
fshift = np.fft.fftshift(f)

# 获取幅值
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 显示傅里叶变换后的二维图像
plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(img, cmap="gray")
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum)
plt.title('Magnitude Image'), plt.xticks([]), plt.yticks([])

# 去掉低频信号
row, col = img.shape
cX, cY = col // 2, row // 2
fshift[(cY - 30):(cY + 30), (cX - 30):(cX + 30)] = 0

# 傅里叶逆变换
fshift_b = np.fft.ifftshift(fshift)
f_b = np.fft.ifft2(fshift_b)
img_back = np.abs(f_b)

# 显示逆变换后的图片
plt.subplot(223), plt.imshow(img_back, cmap="gray")
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()
