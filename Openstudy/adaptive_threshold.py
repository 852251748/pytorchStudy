import cv2
from matplotlib import pyplot as plt

img = cv2.imread("src1.jpg")

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)
ret, Bind_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
Mean_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 2)
Guss_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 9, 2)

titles = ['Original Image', 'Binary Image', 'Mean Image', 'Guss Image']

images = [img, Bind_threshold, Mean_threshold, Guss_threshold]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
