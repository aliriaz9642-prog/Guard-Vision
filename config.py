import os

class Config:
    # System Settings
    SYSTEM_NAME = "AeroGuard Intelligence System"
    VERSION = "2.0.0 (Airport-Grade)"
    DEBUG_MODE = True

    # Camera Settings
    CAMERA_SOURCE = 0  # Change to RTSP URL for IP Camera
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    # Detection & Tracking
    # Detection & Tracking
    YOLO_MODEL_PATH = "yolov8n.pt"
    # Classes: 0=Person, 43=Knife, 76=Scissors. (Note: Standard COCO doesn't have 'gun' usually, but we check for it if user has custom model. 
    # For standard YOLOv8n, we stick to Person(0), Knife(43), Scissors(76), Backpack(24), Handbag(26), Suitcase(28)
    DETECT_CLASSES = [0, 43, 76] 
    WEAPON_CLASSES = [43, 76] # IDs that trigger immediate red alert
    
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    TRACKER_ALGORITHM = "bytetrack" 

    # Face Recognition
    FACE_DB_PATH = os.path.join(os.getcwd(), "data", "suspects")
    AUTHORIZED_DB_PATH = os.path.join(os.getcwd(), "data", "authorized")
    FACE_MATCH_THRESHOLD = 0.55 # Balanced for speed and accuracy
    
    # OPTIMIZATION: Check frequently for new tracks, less frequently for known ones
    FACE_CHECK_INTERVAL_FRAMES = 3 # Hyper-fast (every 3 frames)

    # Behavior Analysis Thresholds
    FPS_ESTIMATE = 30
    LOITERING_TIME_SECONDS = 30  # Increased for realistic airport context
    LOITERING_RADIUS_PITCH = 80 
    PACING_REVERSALS_THRESHOLD = 4
    SUSPICION_DECAY = 0.98 # Slower decay for more persistent tracking
    MAX_HISTORY_POINTS = 500 
    
    # Suspicion Scoring Weights (Now strictly added once or time-scaled)
    SCORE_LOITERING = 20
    SCORE_PACING = 30
    SCORE_WEAPON = 1000 # Instant Max Danger
    SCORE_ABANDONED_OBJECT = 50

    # Privacy
    BLUR_FACES_DEFAULT = False 
    SHOW_IDENTITY_IF_CLEARED = True 

    # Paths
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    SNAPSHOT_DIR = os.path.join(os.getcwd(), "snapshots")

    @staticmethod
    def setup_dirs():
        os.makedirs(Config.FACE_DB_PATH, exist_ok=True)
        os.makedirs(Config.AUTHORIZED_DB_PATH, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.SNAPSHOT_DIR, exist_ok=True)

Config.setup_dirs()
