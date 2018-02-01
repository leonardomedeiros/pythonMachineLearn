import cv2
import numpy as np
img = cv2.imread('../img/louvre.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Church Original', gray)
#Create SIFT Feature Detector object
sift = cv2.SIFT()
keypoints = sift.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))
# Draw rich key points on input image
imageKeyPoints = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Feature Method - SIFT', imageKeyPoints)
cv2.imwrite('../img/louvre_sift_keypoints.jpg',imageKeyPoints)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()