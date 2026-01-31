import logging
import json
import os
from datetime import datetime
from config import Config

class AuditLogger:
    def __init__(self):
        self.log_file = os.path.join(Config.LOG_DIR, f"audit_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Setup standard logger
        self.logger = logging.getLogger("AeroGuard")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)

    def log_event(self, event_type, details):
        """
        Log a structured security event.
        :param event_type: str (e.g., "SUSPICION_ALERT", "SYSTEM_START", "ACCESS_DENIED")
        :param details: dict (contextual info)
        """
        # Custom encoder to handle Numpy/YOLO types like int64
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        payload = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "details": details
        }
        self.logger.info(json.dumps(payload, cls=NpEncoder))

    def log_tracking(self, track_id, location, suspicion_score):
        """High-frequency tracking log (optional, use sparingly)"""
        # self.logger.debug(f"Track {track_id} at {location} | Score: {suspicion_score}")
        pass

# Singleton instance
system_logger = AuditLogger()
