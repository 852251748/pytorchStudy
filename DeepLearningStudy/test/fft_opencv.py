import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("riya.jpg", 0)

f = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(img, cmap="gray")
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Image"), plt.xticks([]), plt.yticks([])

# 去掉高频信号
row, col = img.shape
cx, cy = col // 2, row // 2
zeros = np.zeros((row, col, 2))
zeros[cy - 30:cy + 30, cx - 30:cy + 30] = fshift[cy - 30:cy + 30, cx - 30:cy + 30]

# 傅里叶逆变换
fshift_b = np.fft.ifftshift(zeros)
f_b = cv.idft(np.float32(fshift_b), flags=cv.DFT_COMPLEX_OUTPUT)
img_back = cv.magnitude(f_b[:, :, 0], f_b[:, :, 1])
print(f_b)

plt.subplot(223), plt.imshow(img_back, cmap="gray")
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back)
plt.title("Result in JET"), plt.xticks([]), plt.yticks([])

plt.show()
