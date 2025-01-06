import cv2
from ultralytics import YOLO
from collections import defaultdict

# load the YOLO model
model = YOLO('yolo11l.pt')

class_list = model.names

cap = cv2.VideoCapture('apel.mp4')

# Koordinat garis diagonal
x1_line, y1_line = 21, 238
x2_line, y2_line = 763, 352

# Hitung kemiringan (m) dan y-intercept (c) dari garis
m_line = (y2_line - y1_line) / (x2_line - x1_line)
c_line = y1_line - m_line * x1_line

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of object IDs that have crossed the line
crossed_ids = set()

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  results = model.track(frame, persist=True, classes=[47])
  # print(results)

  if results[0].boxes.data is not None:
    boxes = results[0].boxes.xyxy.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    class_indices = results[0].boxes.cls.int().cpu().tolist()
    confidences = results[0].boxes.conf.cpu()

    cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 3)
    #cv2.putText(frame, 'Red Line', (690, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
      x1, y1, x2, y2 = map(int, box)
      cx = (x1 + x2) // 2 
      cy = (y1 + y2) // 2

      class_name = class_list[class_idx]

      cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

      cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

      # Check if the object has crossed the red line
      y_on_line = m_line * cx + c_line
      if cy > y_on_line and track_id not in crossed_ids:
        # Mark the object as crossed
        crossed_ids.add(track_id)
        class_counts[class_name] += 1
    
    y_offset = 30
    for class_name, count in class_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

  cv2.imshow("YOLO object tracking and counting",frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()