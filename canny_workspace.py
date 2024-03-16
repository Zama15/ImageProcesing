import cv2

img = cv2.imread('imgs/image1.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_2 = cv2.GaussianBlur(img, (5, 5), 0)

canny = cv2.Canny(img_2, 30, 60)
cv2.imshow('Canny', canny)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      video = cv2.GaussianBlur(video, (5, 5), 0)
      cannyVideo = cv2.Canny(video, 30, 60)
      cv2.imshow('Video', cannyVideo)
      '''
        Close the video with the following keys
        Number ASII       Key
        113               q
        81                Q
        120               x
        88                X
        27                ESC
        13                Enter
      '''
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
        break
    else:
      break

cap.release()
cv2.destroyAllWindows()
