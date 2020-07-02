import cv2

img = cv2.imread('9.jpg', 0)
template = cv2.imread('18.jpg', 0)
h, w = template.shape

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

bottom_right = (max_loc[0] + w, max_loc[1] + h)
cv2.rectangle(img, max_loc, bottom_right, 255, 2)

cv2.imshow("img", img)
cv2.waitKey(0)