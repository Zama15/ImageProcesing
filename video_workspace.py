import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print('Error: the video is not opened')
  exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('vids/video1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width,frame_height))

while (True):
  ret, frame = cap.read()
  key = cv2.waitKey(1) & 0xFF
  if ret:
    out.write(frame)
    cv2.imshow('Video', frame)
    out.write(frame)
    if key == ord('q') or key == ord('x') or key == 27 or key == 13:
      break
  else:
    break
  
cap.release()
out.release()
cv2.destroyAllWindows()
