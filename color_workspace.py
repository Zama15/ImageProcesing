import cv2

img_1 = cv2.imread('imgs/image1.jpeg')
img_2 = cv2.resize(img_1, (0, 0), fx=2, fy=2)
# img_3 = cv2.cvtColor(img_2, cv2.COLOR_BGR2YUV)
# img_3 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
img_3 = cv2.cvtColor(img_2, cv2.COLOR_BGR2HSV)

cv2.imshow('Image 3', img_3)
cv2.imshow('Canal 0:', img_3[:,:,0])
cv2.imshow('Canal 1:', img_3[:,:,1])
cv2.imshow('Canal 2:', img_3[:,:,2])
cv2.waitKey(0)