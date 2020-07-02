import cv2
import numpy as np

img = cv2.imread(r"1.jpg")

# cv2.line(img, (100, 30), (210, 180), color=(0, 0, 255), thickness=2)
# cv2.circle(img, (50, 50), 30, (0, 0, 255), 2)
# cv2.rectangle(img, (100, 30), (210, 180), color=(0, 0, 255), thickness=2)
# cv2.ellipse(img, (100, 100), (100, 50), 0, 0, 360, (255, 0, 0), -1)

# pts = np.array([[10, 5], [50, 10], [70, 20], [20, 30]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.polylines(img, [pts], True, (0, 0, 255), 2)

cv2.putText(img, 'beautiful girl', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

cv2.imshow("pic show", img)
cv2.waitKey(0)
