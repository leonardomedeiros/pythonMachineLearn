import cv2
import numpy as np
img = cv2.imread('../img/louvre.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Create FAST detector object
fast = cv2.FastFeatureDetector()
# Create BRIEF extractor object
brief = cv2.DescriptorExtractor_create("BRIEF")
# Determine key points
keypoints = fast.detect(gray, None)
# Obtain descriptors and new final keypoints using BRIEF
keypoints, descriptors = brief.compute(gray, keypoints)
print("Number of keypoints Detected: ", len(keypoints))
# Draw rich keypoints on input image
imgKeyPointsBrief = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Feature Method - BRIEF', imgKeyPointsBrief)
cv2.imwrite('../img/louvre_brief_keypoints.jpg',imgKeyPointsBrief)
cv2.waitKey()
cv2.destroyAllWindows()