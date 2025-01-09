import csv
import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def TakeImage(l1, l2, haar_cascade_path, train_image_path, message, err_screen, text_to_speech):
    if not l1.strip() and not l2.strip():
        t = "Please Enter your Enrollment Number and Name."
        text_to_speech(t)
        message.configure(text=t)
        return
    elif not l1.strip():
        t = "Please Enter your Enrollment Number."
        text_to_speech(t)
        message.configure(text=t)
        return
    elif not l2.strip():
        t = "Please Enter your Name."
        text_to_speech(t)
        message.configure(text=t)
        return

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("Camera not accessible. Please check your webcam.")

        if not os.path.exists(haar_cascade_path):
            raise FileNotFoundError(f"Haar Cascade file not found at {haar_cascade_path}")
        detector = cv2.CascadeClassifier(haar_cascade_path)

        Enrollment = l1.strip()
        Name = l2.strip()
        directory = f"{Enrollment}_{Name}"
        path = os.path.join(train_image_path, directory)

        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Directory created: {path}")
        else:
            raise FileExistsError(f"Directory already exists for {Enrollment}_{Name}")

        sampleNum = 0
        while True:
            ret, img = cam.read()
            if not ret:
                raise Exception("Failed to read frame from camera.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(
                    os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg"),
                    gray[y : y + h, x : x + w],
                )
                cv2.imshow("Frame", img)

            if cv2.waitKey(1) & 0xFF == ord("q") or sampleNum >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        csv_file = "StudentDetails/studentdetails.csv"
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, "a+", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow([Enrollment, Name])

        res = f"Images Saved for ER No: {Enrollment}, Name: {Name}"
        logging.info(res)
        message.configure(text=res)
        text_to_speech(res)

    except FileExistsError as e:
        logging.error(str(e))
        message.configure(text=str(e))
        text_to_speech(str(e))
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        message.configure(text=f"Error: {str(e)}")
        text_to_speech(f"Error: {str(e)}")

# For Test use :


# if __name__ == "__main__":
#     # Paths
#     HAAR_CASCADE_PATH = r"D:\Attendance-Management-system-using-face-recognition-master\Attendance-Management-system-using-face-recognition-master\Attendance-Management-system-using-face-recognition-master\haarcascade_frontalface_default.xml"
#     TRAIN_IMAGE_PATH = "train_images"

#     # Mocked message and text_to_speech functions
#     class MockMessage:
#         def configure(self, text):
#             print(text)

#     def mock_text_to_speech(text):
#         print(f"[TTS]: {text}")

#     message = MockMessage()
#     text_to_speech = mock_text_to_speech

#     # Capture images
#     TakeImage("9027", "Ayush", HAAR_CASCADE_PATH, TRAIN_IMAGE_PATH, message, None, text_to_speech)
