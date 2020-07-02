import cv2 as cv
import numpy as np

img = cv.imread("125.png")

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
ret, thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY_INV)
cv.imshow("leaf1", thresh)
kernel = np.ones((3, 3), np.uint8)
imgmo = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)


sure_bg = cv.dilate(imgmo, kernel, iterations=3)
# sure_bg = cv.dilate(imgmo, kernel)
cv.imshow("leaf", sure_bg)
cv.waitKey(0)
exit()

dist_transform = cv.distanceTransform(imgmo, 1, 5)

_, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv.THRESH_BINARY)

sure_fg = np.uint8(sure_fg)
# print(sure_bg.shape, type(sure_bg[0][0]), type(sure_fg[0][0]))
unknown = cv.subtract(sure_bg, sure_fg)

_, markers = cv.connectedComponents(sure_fg)

markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markerste = np.uint8(markers)
markerste = cv.convertScaleAbs(markerste, 0, 20)
cv.imshow("markerste", markerste)

markers = cv.watershed(img, markers)
markerste = np.uint8(markers)
# markerste = cv.convertScaleAbs(markerste, 0, 20)
cv.imshow("markers", markerste)
img[markers == -1] = [0, 0, 255]
cv.imshow("img", img)

# kernel1 = np.ones((10, 10), np.uint8)
# sure_fg1 = cv.erode(imgmo, kernel1, iterations=3)
#
# unknown1 = cv.subtract(sure_bg, sure_fg1)
# _, markers2 = cv.connectedComponents(sure_fg1)
# markers1 = markers2 + 1
# # Now, mark the region of unknown with zero
# markers1[unknown1 == 255] = 0
# markers1 = cv.watershed(img, markers1)
# img[markers1 == -1] = [0, 255, 0]
#
# cv.imshow("img1", img)
cv.waitKey(0)
