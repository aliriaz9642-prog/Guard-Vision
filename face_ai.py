import cv2
import numpy as np
import insightface
from config import Config
from logger import system_logger

class FaceAI:
    def __init__(self):
        self.model = None
        self.is_ready = False

    def initialize(self):
        """Lazy load the model to speed up startup until needed."""
        if self.is_ready: return

        try:
            # ctx_id=0 for GPU, -1 for CPU. usage depends on env.
            # safe fallback to CPU if GPU fails would be nice, but insightface handles ctx_id
            ctx = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1
            self.model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.model.prepare(ctx_id=ctx, det_size=(640, 640))
            self.is_ready = True
            system_logger.logger.info(f"FaceAI Model initialized on ctx={ctx}")
        except Exception as e:
            system_logger.logger.error(f"Failed to init FaceAI: {e}")

    def get_face(self, frame_crop):
        """
        Returns the embedding of the largest face in the crop.
        """
        if not self.is_ready: self.initialize()
        
        faces = self.model.get(frame_crop)
        if not faces:
            return None
        
        # Return the largest face (center main subject)
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].embedding

    def scan_frame(self, frame):
        """
        Returns all faces in a frame (for full scan).
        """
        if not self.is_ready: self.initialize()
        return self.model.get(frame)

face_recognition = FaceAI()
