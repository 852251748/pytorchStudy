import cv2 as cv

image = cv.imread(r"riya.jpg")
image = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
print(image.shape)
cv.imshow("pic show", image)
cv.waitKey(0)





