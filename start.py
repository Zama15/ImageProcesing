import cv2

img = cv2.imread('imgs/image1.jpeg')
img_2 = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)

cv2.imshow('Image', img_2)
cv2.waitKey(0)