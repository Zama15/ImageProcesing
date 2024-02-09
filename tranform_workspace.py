import cv2
import numpy as np

img = cv2.imread('imgs/image1.jpeg')
num_rows, num_cols = img.shape[:2]

'''
  Translation
  It move the image to the right and down
  The translation matrix is:
  [1, 0, Tx]
  [0, 1, Ty]
  where Tx is the number of pixels to move the image to the right
  and Ty is the number of pixels to move the image down
'''
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
cv2.imshow('Translation', img_translation)


'''
  Rotation
  The rotation matrix is:
  [cos(theta), -sin(theta)]
  [sin(theta), cos(theta)]
  where theta is the angle of rotation
'''
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)


'''
  Video
    Get the video from the webcam, get the shape of the frame
    - Translation
      Apply the translation matrix to the frame
      Convert the frame to YUV
    - Rotation
      Apply the rotation matrix to the frame
      Convert the frame to HSV
    Show the frames
    If the user press 'q' or 'x' or 'esc' or 'enter' the program ends
'''
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
  ret, frame = cap.read()
  num_rows, num_cols = frame.shape[:2]
  if ret:
    video_translation = cv2.warpAffine(frame, translation_matrix, (num_cols, num_rows))
    video_translation = cv2.cvtColor(video_translation, cv2.COLOR_BGR2YUV)
    cv2.imshow('Translation Video', video_translation)

    video_rotation = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))
    video_rotation = cv2.cvtColor(video_rotation, cv2.COLOR_BGR2HSV)
    cv2.imshow('Rotation Video', video_rotation)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
      break
  else:
    break

cap.release()
cv2.destroyAllWindows()