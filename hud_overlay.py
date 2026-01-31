import cv2
import numpy as np
from config import Config
import time

class HUD:
    
    # Colors (BGR)
    COLOR_SAFE = (0, 255, 128)      # Spring Green
    COLOR_WARN = (0, 215, 255)      # Gold
    COLOR_DANGER = (0, 0, 255)      # Red
    COLOR_TEXT = (255, 255, 255)    # White
    COLOR_HUD = (255, 191, 0)       # Deep Sky Blue
    
    @staticmethod
    def draw(frame, active_track_ids, state_manager):
        """
        Draws the "Airport Intelligence" overlay.
        """
        h, w = frame.shape[:2]
        
        # 1. Screen-wide Threat Alert (Flash)
        has_threat = any(p.role == "Suspect" or p.suspicion_score > 80 for p in state_manager.active_tracks.values())
        if has_threat and int(time.time() * 2) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), 10) # Thick red border
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.putText(frame, "!!! SECURITY THREAT DETECTED !!!", (w//2 - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # 2. Draw Weapons (High Priority)
        for (box, label, track_id) in state_manager.detected_weapons:
            x1, y1, x2, y2 = map(int, box)
            HUD._draw_fancy_box(frame, (x1, y1, x2, y2), HUD.COLOR_DANGER, label=f"THREAT: {label}", thickness=3)
            # Flash effect
            if int(time.time() * 5) % 2 == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), HUD.COLOR_DANGER, -1)
                cv2.addWeighted(frame[y1:y2, x1:x2], 0.7, np.full((y2-y1, x2-x1, 3), HUD.COLOR_DANGER, dtype=np.uint8), 0.3, 0, frame[y1:y2, x1:x2])

        # 3. Draw Person Overlays
        for tid in active_track_ids:
            if tid not in state_manager.active_tracks: continue
            
            person = state_manager.active_tracks[tid]
            x1, y1, x2, y2 = map(int, person.bbox)
            
            # Privacy
            if Config.BLUR_FACES_DEFAULT and person.suspicion_score < 50:
                HUD._blur_face(frame, x1, y1, x2, y2)

            # Color & Status Logic
            color = HUD.COLOR_SAFE
            status = "VISITOR"
            
            if person.role == "Staff":
                color = (0, 255, 0) # Pure Green for Authorized
                status = "AUTHORIZED PERSONNEL"
            elif person.role == "Suspect" or person.suspicion_score > 80:
                color = HUD.COLOR_DANGER
                status = "THREAT DETECTED"
            elif person.suspicion_score > 40:
                color = HUD.COLOR_WARN
                status = "SUSPICIOUS"

            # Identity Logic
            identity_text = "SCANNING..."
            if person.is_identified:
                if person.role == "Staff":
                    identity_text = f"WELCOME, {person.name.upper()}"
                else:
                    identity_text = f"ID: {person.name.upper()} [{person.role}]"
            
            # Draw Box
            HUD._draw_fancy_box(frame, (x1, y1, x2, y2), color, label=status)

            # Draw Info Panel
            HUD._draw_info_panel(frame, x1, y1, x2, identity_text, person.suspicion_score, person.active_alerts, color)

        # 4. Status Bar
        HUD._draw_status_bar(frame)

    @staticmethod
    def _draw_fancy_box(frame, bbox, color, label=None, thickness=2):
        x1, y1, x2, y2 = bbox
        
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Corner Brackets
        l = min(int((x2-x1)*0.2), 30)
        # Top-Left
        cv2.line(frame, (x1, y1), (x1+l, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1+l), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2-l, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1+l), color, thickness)
        # Bot-Left
        cv2.line(frame, (x1, y2), (x1+l, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2-l), color, thickness)
        # Bot-Right
        cv2.line(frame, (x2, y2), (x2-l, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2-l), color, thickness)

        if label:
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1-20), (x1+t_size[0]+10, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    @staticmethod
    def _draw_info_panel(frame, x1, y1, x2, identity, score, alerts, color):
        # Draw side or top panel
        # Let's put it on top of the box if space, or float
        
        # Suspicion Bar
        bar_w = x2 - x1
        bar_h = 4
        fill_w = int((score / 100) * bar_w)
        cv2.rectangle(frame, (x1, y1-25), (x1 + bar_w, y1-21), (50,50,50), -1)
        cv2.rectangle(frame, (x1, y1-25), (x1 + fill_w, y1-21), color, -1)
        
        # Identity
        cv2.putText(frame, identity, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Alerts
        if alerts:
            y_off = y1 - 45
            for alert in alerts:
                cv2.putText(frame, f"/!\\ {alert}", (x1, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                y_off -= 15

    @staticmethod
    def _draw_vignette(frame):
        # Subtle darkening of corners
        pass # Optimization: Skip for FPS, or implement lightweight version if requested

    @staticmethod
    def _draw_status_bar(frame):
        h, w = frame.shape[:2]
        # Bottom Bar
        cv2.rectangle(frame, (0, h-30), (w, h), (10,10,10), -1)
        
        # Tech Text
        text = f"SYSTEM: {Config.SYSTEM_NAME} | ACTIVE_MONITORING | V{Config.VERSION}"
        cv2.putText(frame, text, (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        
        # Time
        time_str = time.strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (w-100, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

    @staticmethod
    def _blur_face(frame, x1, y1, x2, y2):
        face_h = int((y2 - y1) * 0.2)
        fy2 = y1 + face_h
        h, w = frame.shape[:2]
        x1, y1, x2, fy2 = max(0, x1), max(0, y1), min(w, x2), min(h, fy2)
        if (x2-x1) > 0 and (fy2-y1) > 0:
            roi = frame[y1:fy2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (51, 51), 30)
            frame[y1:fy2, x1:x2] = blurred

