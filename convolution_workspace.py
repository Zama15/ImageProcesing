import cv2
import numpy as np

img_1 = cv2.imread('imgs/image1.jpeg')
rows, cols = img_1.shape[:2]

kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
kernel_3x3 = np.ones((3, 3), np.float32) / 9.0
kernel_5x5 = np.ones((5, 5), np.float32) / 25.0


cv2.imshow('Image 1', img_1)

output_kernel = cv2.filter2D(img_1, -1, kernel_identity)
output_3x3 = cv2.filter2D(img_1, -1, kernel_3x3)
output_5x5 = cv2.filter2D(img_1, -1, kernel_5x5)

cv2.imshow('Identity filter', output_kernel)
cv2.imshow('3x3 filter', output_3x3)
cv2.imshow('5x5 filter', output_5x5)

cv2.waitKey(0)