import cv2
import numpy as np
img = cv2.imread('../img/louvre.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Create SURF Feature Detector object
surf = cv2.SURF()
# Only features, whose hessian is larger than hessianThreshold are retained by the detector
surf.hessianThreshold = 500
keypoints, descriptors = surf.detectAndCompute(gray, None)
print("Number of keypoints Detected: ", len(keypoints))
# Draw rich key points on input image
imageKeyPoints = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Feature Method - SURF', imageKeyPoints)
cv2.imwrite('../img/louvre_surf_keypoints.jpg',imageKeyPoints)
cv2.waitKey()
cv2.destroyAllWindows()