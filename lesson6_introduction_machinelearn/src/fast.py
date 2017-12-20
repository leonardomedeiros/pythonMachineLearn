import cv2
import numpy as np
img = cv2.imread('../img/louvre.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Create FAST Detector object
fast = cv2.FastFeatureDetector()
# Obtain Key points, by default non max suppression is On
# to turn off set fast.setBool('nonmaxSuppression', False)
keypoints = fast.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))
# Draw rich keypoints on input image
imageKeyPoints = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Feature Method - FAST', imageKeyPoints)
cv2.imwrite('../img/louvre_fast_keypoints.jpg',imageKeyPoints)
cv2.waitKey()
cv2.destroyAllWindows()