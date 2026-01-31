import os
import cv2
import numpy as np
import pickle
from config import Config
from logger import system_logger

class Database:
    def __init__(self):
        self.suspects = {} # {name: embedding}
        self.metadata = {} # {name: {role: "Suspect", notes: "..."}}
        self.load_database()

    def load_database(self):
        """Loads all .npy embeddings and metadata from different categories."""
        # Ensure directories exist
        Config.setup_dirs()
        
        from face_ai import face_recognition
        face_recognition.initialize()

        categories = [
            (Config.FACE_DB_PATH, "Suspect"),
            (Config.AUTHORIZED_DB_PATH, "Staff")
        ]

        # 1. Auto-Ingest Images from both folders
        for folder, role in categories:
            for file in os.listdir(folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(file)[0]
                    npy_path = os.path.join(folder, f"{name}.npy")
                    
                    if not os.path.exists(npy_path):
                        system_logger.logger.info(f"Generating embedding for {role}: {file}...")
                        img_path = os.path.join(folder, file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            emb = face_recognition.get_face(img)
                            if emb is not None:
                                self.add_person(name, emb, folder, role)
                            else:
                                system_logger.logger.warning(f"No face found in {file}")

        # 2. Load Embeddings into memory
        count = 0
        for folder, role in categories:
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    name = os.path.splitext(file)[0]
                    embedding_path = os.path.join(folder, file)
                    try:
                        self.suspects[name] = np.load(embedding_path)
                        self.metadata[name] = {"role": role, "risk": "High" if role == "Suspect" else "None"} 
                        count += 1
                    except Exception as e:
                        system_logger.logger.error(f"Failed to load {name}: {e}")
        
        system_logger.logger.info(f"Database loaded with {count} individuals across all categories.")

    def add_person(self, name, embedding, folder, role):
        """Save embedding and update memory."""
        path = os.path.join(folder, f"{name}.npy")
        np.save(path, embedding)
        self.suspects[name] = embedding
        self.metadata[name] = {"role": role}
        system_logger.log_event("ENTRY_ADDED", {"name": name, "role": role})

    def match_face(self, target_embedding, threshold=0.5):
        """
        Compare target_embedding against all suspects.
        Returns: (name, metadata, similarity_score) or (None, None, 0)
        """
        best_match = None
        best_score = -1

        target_norm = np.linalg.norm(target_embedding)

        for name, db_emb in self.suspects.items():
            db_norm = np.linalg.norm(db_emb)
            # Cosine similarity
            sim = np.dot(target_embedding, db_emb) / (target_norm * db_norm)
            
            if sim > best_score:
                best_score = sim
                best_match = name

        if best_score > threshold:
            return best_match, self.metadata.get(best_match, {}), best_score
        
        return None, None, 0

# Singleton
suspect_db = Database()
