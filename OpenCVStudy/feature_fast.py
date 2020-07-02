import cv2

src = cv2.imread("33.jpg")
grayImg = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create(threshold=35)
# fast.setNonmaxSuppression(False)
kp = fast.detect(grayImg, None)
img2 = cv2.drawKeypoints(src, kp, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print('Threshold: ', fast.getThreshold())
print('nonmaxSuppression: ', fast.getNonmaxSuppression())
print('neighborhood: ', fast.getType())
print('Total Keypoints with nonmaxSuppression: ', len(kp))
#
cv2.imshow('fast_true', img2)
#
# fast.setNonmaxSuppression(False)
# kp = fast.detect(grayImg, None)
#
# print('Total Keypoints without nonmaxSuppression: ', len(kp))
#
# img3 = cv2.drawKeypoints(src, kp, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow('fast_false', img3)

cv2.waitKey()