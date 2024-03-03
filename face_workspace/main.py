import cv2
import numpy as np
from os import path

current_dir = path.dirname(path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
left_ear_cascade = cv2.CascadeClassifier(path.join(current_dir, 'haarcascade_mcs_leftear.xml'))
right_ear_cascade = cv2.CascadeClassifier(path.join(current_dir, 'haarcascade_mcs_rightear.xml'))

def file_error(file, file_name):
    if file.empty():
        raise IOError('Unable to load the ' + file_name + ' cascade classifier xml file')

file_error(face_cascade, 'face')
file_error(eye_cascade, 'eye')
file_error(right_ear_cascade, 'right ear')
file_error(left_ear_cascade, 'left ear')

cap = cv2.VideoCapture(0)
ds_factor = 1.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_var = (gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(*detect_var)
    left_ear = left_ear_cascade.detectMultiScale(*detect_var)
    right_ear = right_ear_cascade.detectMultiScale(*detect_var)
    thickness = 2
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cv2.putText(frame, 'Face', (x, y), font, 1, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        for(x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3*(w_eye + h_eye))
            cv2.putText(roi_color, 'Eye', (x_eye, y_eye), font, 1, color, 2, cv2.LINE_AA)
            cv2.circle(roi_color, center, radius, color, thickness)

    for(x_ear_l, y_ear_l, w_ear_l, h_ear_l) in left_ear:
        center_ear_l = (int(x_ear_l + 0.5*w_ear_l), int(y_ear_l + 0.5*h_ear_l))
        radius_ear_l = int(0.3*(w_ear_l + h_ear_l))
        cv2.putText(frame, 'Left ear', (x_ear_l, y_ear_l), font, 1, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center_ear_l, radius_ear_l, color, thickness)

    for(x_ear_r, y_ear_r, w_ear_r, h_ear_r) in right_ear:
        center_ear_r = (int(x_ear_r + 0.5*w_ear_r), int(y_ear_r + 0.5*h_ear_r))
        radius_ear_r = int(0.3*(w_ear_r + h_ear_r))
        cv2.putText(frame, 'Right ear', (x_ear_r, y_ear_r), font, 1, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center_ear_r, radius_ear_r, color, thickness)
        
            
    cv2.imshow('Decter of face and eyes', frame)
    c = cv2.waitKey(1)
    if c == 27 or c == ord('q') or c == 13 or c == 27 or c == 32:
        break

cap.release()
cv2.destroyAllWindows()
