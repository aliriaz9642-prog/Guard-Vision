from person import Person
from behavior_engine import BehaviorEngine
from logger import system_logger
from config import Config
import alert

class StateManager:
    def __init__(self):
        self.active_tracks = {} # {track_id: Person}
        self.detected_weapons = [] # List of tuples (box, label, track_id)

    def update_tracks(self, yolov8_results):
        """
        Sync active_tracks with YOLOv8 tracking results.
        """
        current_frame_ids = []
        self.detected_weapons = [] # Reset per frame
        
        if yolov8_results[0].boxes.id is not None:
            # We have tracking IDs
            boxes = yolov8_results[0].boxes.xyxy.cpu().numpy()
            track_ids = yolov8_results[0].boxes.id.cpu().numpy().astype(int)
            classes = yolov8_results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                
                # Check for Weapons
                if cls in Config.WEAPON_CLASSES:
                    label = "Unidentified Object"
                    if cls == 43: label = "KNIFE"
                    elif cls == 76: label = "SCISSORS" # or generic "SHARP OBJECT"
                    
                    self.detected_weapons.append((box, label, track_id))
                    
                    # TRIGGER ALERT
                    system_logger.log_event("WEAPON_DETECTED", {"type": label, "track_id": int(track_id)})
                    alert.red_alert_sound()
                    continue # Don't treat as a person

                # Only process Persons (Class 0)
                if cls == 0:
                    current_frame_ids.append(track_id)
                    
                    # Create or Update Person
                    if track_id not in self.active_tracks:
                        self.active_tracks[track_id] = Person(track_id, box)
                        system_logger.log_event("PERSON_ENTERED", {"track_id": int(track_id)})
                    else:
                        self.active_tracks[track_id].update(box)

                    # Run Behavior Analysis
                    BehaviorEngine.analyze(self.active_tracks[track_id])
                    
                    # Check for Suspicion Alert
                    if self.active_tracks[track_id].suspicion_score > 80:
                         alert.red_alert_sound()

        
        return self.active_tracks
