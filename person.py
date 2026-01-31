import time
import numpy as np
from collections import deque
from config import Config

class Person:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.centroid = self._calculate_centroid(bbox)
        
        # Identity
        self.face_embedding = None
        self.name = "Unknown"
        self.role = "Visitor" # Visitor, Staff, Suspect
        self.is_identified = False
        self.last_face_check_time = 0
        
        # Behavior History
        self.location_history = deque(maxlen=Config.MAX_HISTORY_POINTS)
        self.velocity_history = deque(maxlen=30)
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        
        # State
        self.suspicion_score = 0
        self.active_alerts = []  # List of strings e.g. ["Loitering", "Pacing"]
        self.movement_state = "Standing" # Standing, Walking, Running, Pacing
        
        # Update history init
        self.location_history.append((self.centroid, self.last_seen_time))

    def _calculate_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox):
        self.bbox = bbox
        new_centroid = self._calculate_centroid(bbox)
        current_time = time.time()
        
        # Calculate instant velocity (pixels per second)
        dt = current_time - self.last_seen_time
        if dt > 0:
            dx = np.linalg.norm(np.array(new_centroid) - np.array(self.centroid))
            velocity = dx / dt
            self.velocity_history.append(velocity)
        
        self.centroid = new_centroid
        self.last_seen_time = current_time
        self.location_history.append((self.centroid, current_time))
        
        # Decay suspicion slowly if no active alerts
        if not self.active_alerts and self.suspicion_score > 0:
            self.suspicion_score *= Config.SUSPICION_DECAY

    def set_identity(self, name, role, score=0):
        self.name = name
        self.role = role
        self.is_identified = True
        
        if role == "Suspect":
            self.suspicion_score = 100
            if f"Identified Suspect: {name}" not in self.active_alerts:
                self.active_alerts.append(f"Identified Suspect: {name}")
        elif role == "Staff":
            # Reward Staff: Auto-clear many minor suspicions
            self.suspicion_score = 0
            self.active_alerts = [a for a in self.active_alerts if "Weapon" in a] # Only keep weapon alerts for staff
            self.last_staff_check = time.time()

    def add_suspicion(self, points, reason):
        # Only add points if it's a new alert or significant
        if reason not in self.active_alerts:
            self.active_alerts.append(reason)
            self.suspicion_score = min(100, self.suspicion_score + points)
        else:
            # If already alerting, slowly increase (prevent instant 100)
            self.suspicion_score = min(100, self.suspicion_score + (points * 0.05)) 

    def clear_alerts(self, specific_alert=None):
        if specific_alert:
            if specific_alert in self.active_alerts:
                self.active_alerts.remove(specific_alert)
        else:
            self.active_alerts = []

    @property
    def age_on_camera(self):
        return time.time() - self.first_seen_time
