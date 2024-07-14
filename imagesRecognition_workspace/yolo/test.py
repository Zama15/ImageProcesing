from os import path
import yolov5
import cv2

DIR = path.dirname(path.abspath(__file__))

# load pretrained model
model = yolov5.load('yolov5s.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set video
img = 'vids/race_car.mp4'

# read video
cap = cv2.VideoCapture(img)

while True:
  ret, frame = cap.read()

  if not ret:
    break

  # perform inference
  results = model(frame)
  
  # parse results
  predictions = results.pred[0]
  boxes = predictions[:, :4] # x1, y1, x2, y2
  categories = predictions[:, 5]
  
  for box, category in zip(boxes, categories):
    if int(category) == 2:
      x1, y1, x2, y2 = [int(i) for i in box]
      cv2.putText(frame, f'car', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

  cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

