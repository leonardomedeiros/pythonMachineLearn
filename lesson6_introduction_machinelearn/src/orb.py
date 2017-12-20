import cv2
import numpy as np
img = cv2.imread('../img/louvre.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Create ORB object, we can specify the number of key points we desire
orb = cv2.ORB()
# Determine key points
keypoints = orb.detect(gray, None)
# Obtain the descriptors
keypoints, descriptors = orb.compute(gray, keypoints)
print("Number of keypoints Detected: ", len(keypoints))
# Draw rich keypoints on input img
imgOrbKeyPoints = cv2.drawKeypoints(img, keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Feature Method - ORB', imgOrbKeyPoints)
cv2.imwrite('../img/louvre_orb_keypoints.jpg',imgOrbKeyPoints)
cv2.waitKey()
cv2.destroyAllWindows()