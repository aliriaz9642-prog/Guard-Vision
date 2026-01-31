import cv2
import time
from ultralytics import YOLO
from config import Config
from logger import system_logger
from state_manager import StateManager
from hud_overlay import HUD
from database import suspect_db
from face_ai import face_recognition

def main():
    print(f"Starting {Config.SYSTEM_NAME}...")
    system_logger.log_event("SYSTEM_START", {"version": Config.VERSION})

    # 1. Initialize Models
    print("Loading Detection Model...")
    model = YOLO(Config.YOLO_MODEL_PATH)
    
    print("Loading Face Recognition...")
    face_recognition.initialize() 

    # 2. Initialize Camera
    cap = cv2.VideoCapture(Config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # 3. State Managers
    state_manager = StateManager()

    print("System Online. Press 'ESC' to exit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # A. TRACKING (Detection + ID)
        results = model.track(frame, persist=True, verbose=False, classes=Config.DETECT_CLASSES)

        # B. UPDATE STATE (Logic)
        current_active_ids = []
        if results[0].boxes.id is not None:
            state_manager.update_tracks(results)
            current_active_ids = [tid for tid in state_manager.active_tracks if tid in results[0].boxes.id.cpu().numpy()]

        # C. BEHAVIOR & IDENTITY REFINEMENT
        faces_checked = 0
        MAX_FACES_PER_FRAME = 3 # Increased sensing speed
        
        active_people = [state_manager.active_tracks[tid] for tid in current_active_ids]
        active_people.sort(key=lambda p: p.last_face_check_time)

        for person in active_people:
            tid = person.track_id
            should_check = False
            
            if not person.is_identified:
                should_check = True
            elif (frame_count % Config.FACE_CHECK_INTERVAL_FRAMES == 0) and faces_checked < MAX_FACES_PER_FRAME:
                if (time.time() - person.last_face_check_time) > 1.0: 
                     should_check = True

            if should_check and faces_checked < MAX_FACES_PER_FRAME:
                faces_checked += 1
                person.last_face_check_time = time.time()
                
                x1, y1, x2, y2 = map(int, person.bbox)
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    emb = face_recognition.get_face(face_crop)
                    if emb is not None:
                        name, meta, score = suspect_db.match_face(emb, Config.FACE_MATCH_THRESHOLD)
                        if name:
                            role = meta.get("role", "Unknown")
                            person.set_identity(name, role, score)
                            system_logger.log_event("IDENTIFIED", {"track_id": int(tid), "name": name, "role": role})
                            
                            # CEO FIX: Trigger alert sound IMMEDIATELY for suspects
                            if role == "Suspect":
                                import alert
                                alert.red_alert_sound()

        # D. VISUALIZATION (Overlay)
        HUD.draw(frame, current_active_ids, state_manager)

        # Show Output
        cv2.imshow(Config.SYSTEM_NAME, frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('r'): # Reset
            state_manager = StateManager()
            print("System Reset")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    system_logger.log_event("SYSTEM_SHUTDOWN", {})
    print("System Shutdown.")

if __name__ == "__main__":
    main()
