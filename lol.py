import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_pupil(eye):
  gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] != 0:
      cx = int(M["m10"] / M["m00"])
      cy = int(M["m01"] / M["m00"])
      return (cx, cy)
  return None

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX 

while True:
  ret, frame = cap.read()
  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
    cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for (ex, ey, ew, eh) in eyes:
      cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) 
      pupil_position = detect_pupil(roi_color[ey:ey+eh, ex:ex+ew])
      if pupil_position:
        cv2.circle(roi_color, (ex+pupil_position[0], ey+pupil_position[1]), 3, (0, 0, 255), -1)
        cv2.putText(roi_color, "Eye with Pupil", (ex, ey-5), font, 0.5, (0, 0, 255), 2)

    for (sx, sy, sw, sh) in smiles:
      cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
      cv2.putText(roi_color, "Smile", (sx, sy-5), font, 0.5, (0, 0, 255), 2)

  cv2.imshow('Face, Eye and Smile Detection', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
