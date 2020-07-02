import cv2


img = cv2.imread("riya.jpg")
# cv2.line(img, (100, 30), (210, 180), color=(0, 0, 245), thickness=2)
# cv2.circle(img, (160, 100), 50, color=(100, 200, 230), thickness=2)
# cv2.rectangle(img, (100, 30), (210, 180), color=(100, 200, 100), thickness=2)
# cv2.ellipse(img, (100, 100), (100, 50), 0, 0, 360, (255, 0, 0), 2)
#cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "beautiful girl", (35, 128), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, color=(255, 0, 0))
cv2.imshow("img", img)
cv2.waitKey(0)