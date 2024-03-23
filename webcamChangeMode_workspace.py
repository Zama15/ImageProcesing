import cv2

cap = cv2.VideoCapture(0)
state = 0

def modidy_frame(frame, state):
  if state == 0:
    return frame
  elif state == 1:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  elif state == 2:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
  elif state == 3:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

while(cap.isOpened()):
  ret, frame = cap.read()

  modified_frame = modidy_frame(frame, state)
  cv2.imshow('Modified Frame', modified_frame)

  key = cv2.waitKey(1) & 0xFF

  if key in [ord('n'), ord('N')]:
    state = 0
  elif key in [ord('g'), ord('G')]:
    state = 1
  elif key in [ord('y'), ord('Y')]:
    state = 2
  elif key in [ord('h'), ord('H')]:
    state = 3
  elif key in [27, 13, ord('q'), ord('Q'), ord('x'), ord('X')]:
    break

cap.release()
cv2.destroyAllWindows()
