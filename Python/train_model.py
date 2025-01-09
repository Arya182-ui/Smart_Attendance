import os
import cv2
import numpy as np
from PIL import Image

haarcascade_path = r"haarcascade_frontalface_default.xml"
train_image_path = r"trainimage"  
model_save_path = r"./TrainingImageLabel/Trainner.yml"
if not os.path.exists('./TrainingImageLabel'):
    os.makedirs('./TrainingImageLabel')


recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists(haarcascade_path):
    print("Haar Cascade file not found")
    exit()

detector = cv2.CascadeClassifier(haarcascade_path)

def get_images_and_labels(path):
    faces = []
    ids = []
    
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, file)
            try:
                pil_image = Image.open(image_path).convert("L")  
                image_np = np.array(pil_image, "uint8")

                person_id = int(os.path.splitext(file)[0].split("_")[1])

                faces.append(image_np)
                ids.append(person_id)
            except Exception as e:
                print(f"Skipping file {file}: {e}")

    return faces, ids


faces, ids = get_images_and_labels(train_image_path)


recognizer.train(faces, np.array(ids))


recognizer.save(model_save_path)
print(f"Model saved to {model_save_path}")
