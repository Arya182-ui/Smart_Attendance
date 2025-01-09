import os
import cv2
import numpy as np
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def TrainImage(haar_cascade_path, image_path, model_save_path, message, text_to_speech):
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        if not os.path.exists(haar_cascade_path):
            raise FileNotFoundError(f"Haar Cascade file not found at {haar_cascade_path}")

        detector = cv2.CascadeClassifier(haar_cascade_path)
        
        faces, Ids = getImagesAndLabels(image_path)

        if not faces:
            raise ValueError("No valid faces found for training.")

        recognizer.train(faces, np.array(Ids))
        recognizer.save(model_save_path)

        res = f"Training complete. {len(faces)} faces trained for {len(set(Ids))} unique IDs."
        logging.info(res)
        message.configure(text=res)
        text_to_speech(res)
    except Exception as e:
        error_message = f"Error in training: {str(e)}"
        logging.error(error_message)
        message.configure(text=error_message)
        text_to_speech(error_message)

def getImagesAndLabels(path):
    faces = []
    Ids = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            imagePath = os.path.join(subdir_path, file)
            try:
                pilImage = Image.open(imagePath).convert("L") 
                imageNp = np.array(pilImage, "uint8")

                # Extract ID from filename
                Id = int(os.path.splitext(file)[0].split("_")[1])

                faces.append(imageNp)
                Ids.append(Id)
            except Exception as e:
                logging.warning(f"Skipping file {file}: {e}")

    return faces, Ids

if __name__ == "__main__":
    HAAR_CASCADE_PATH = r"haarcascade_frontalface_default.xml"
    TRAIN_IMAGE_PATH = r"./trainimage"
    if not os.path.exists(TRAIN_IMAGE_PATH):
        os.makedirs(TRAIN_IMAGE_PATH)
        logging.info(f"Created missing directory: {TRAIN_IMAGE_PATH}")
    MODEL_SAVE_PATH = "trained_model.yml"

    class MockMessage:
        def configure(self, text):
            print(text)

    def mock_text_to_speech(text):
        print(f"[TTS]: {text}")

    message = MockMessage()
    text_to_speech = mock_text_to_speech

    TrainImage(HAAR_CASCADE_PATH, TRAIN_IMAGE_PATH, MODEL_SAVE_PATH, message, text_to_speech)
