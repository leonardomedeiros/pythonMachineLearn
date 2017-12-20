import cv2
import numpy as np
image = cv2.imread('../img/opencv_inv.png', 0)
cv2.imshow('Original', image)
cv2.waitKey(0)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(image, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
dilation = cv2.dilate(image, kernel, iterations = 1)
cv2.imshow('Dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()