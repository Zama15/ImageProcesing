import cv2
import numpy as np

'''
Bitwise operations are used in image processing to perform image masking, convolutions, and other operations.\

Bitwise AND
The bitwise AND operation is used to keep only the pixels that are common in both images.

Bitwise OR
The bitwise OR operation is used to keep the pixels that are present in either of the images.

Bitwise XOR
The bitwise XOR operation is used to keep the pixels that are present in one of the images, but not in both.

Bitwise NOT
The bitwise NOT operation is used to invert the pixel values.
'''
# Create a square image
img1 = np.zeros((400, 600), dtype=np.uint8)
img1[100:300, 200:400] = 255
cv2.imshow('Artificial Image 1', img1)
# Create a circular image
img2 = np.zeros((400, 600), dtype=np.uint8)
img2 = cv2.circle (img2, (300, 200), 100, 255, -1)
cv2.imshow('Artificial Image 2', img2)
# Perform bitwise AND operation
bitwise_and = cv2.bitwise_and(img1, img2)
cv2.imshow('Bitwise AND', bitwise_and)
# Perform bitwise OR operation
bitwise_or = cv2.bitwise_or(img1, img2)
cv2.imshow('Bitwise OR', bitwise_or)
# Perform bitwise XOR operation
bitwise_xor = cv2.bitwise_xor(img1, img2)
cv2.imshow('Bitwise XOR', bitwise_xor)
# Perform bitwise NOT operation
bitwise_not1 = cv2.bitwise_not(img1)
bitwise_not2 = cv2.bitwise_not(img2)
cv2.imshow('Bitwise NOT 1', bitwise_not1)
cv2.imshow('Bitwise NOT 2', bitwise_not2)

cv2.waitKey(0)
cv2.destroyAllWindows()
