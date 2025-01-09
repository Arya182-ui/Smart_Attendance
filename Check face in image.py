import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"D:\Attendance-Management-system-using-face-recognition-master\TrainingImageLabel\Trainner.yml")

detector = cv2.CascadeClassifier(r"D:\Attendance-Management-system-using-face-recognition-master\Attendance-Management-system-using-face-recognition-master\Attendance-Management-system-using-face-recognition-master\haarcascade_frontalface_default.xml")

attendance_log = []

def log_attendance(id_):
    if id_ not in attendance_log:
        attendance_log.append(id_)
        print(f"Attendance marked for ID: {id_}")
    else:
        print(f"ID {id_} already present.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  
        id_, confidence = recognizer.predict(face) 

        if confidence < 100:  
            print(f"Face recognized! ID: {id_}, Confidence: {confidence}")
            log_attendance(id_)
        else:
            print("Unknown face detected.")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
