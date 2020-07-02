import cv2

img = cv2.imread("33.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

kp = orb.detect(grayImg, None)
kp, des = orb.compute(grayImg, kp)

img2 = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255), flags=0)

cv2.imshow("img", img2)
cv2.waitKey(0)
