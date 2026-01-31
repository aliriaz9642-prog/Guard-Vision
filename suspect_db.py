import os
import cv2
import numpy as np
from deepface import DeepFace

DB_PATH = "data/suspects"
os.makedirs(DB_PATH, exist_ok=True)

def add_suspect(name, image_path):
    img = cv2.imread(image_path)
    embedding = DeepFace.represent(img, model_name="Facenet")[0]["embedding"]
    np.save(f"{DB_PATH}/{name}.npy", embedding)
    print(f"Suspect {name} added.")

def load_suspects():
    data = {}
    for file in os.listdir(DB_PATH):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            data[name] = np.load(f"{DB_PATH}/{file}")
    return data

