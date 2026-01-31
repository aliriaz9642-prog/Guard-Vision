import numpy as np
from config import Config

class BehaviorEngine:
    @staticmethod
    def analyze(person):
        """
        Analyzes the person's history to detect anomalies.
        Updates person.active_alerts and person.suspicion_score.
        """
        # 1. Loitering Detection
        if person.role != "Staff": # Authorized personnel are allowed to stand/work
            if BehaviorEngine.check_loitering(person):
                person.add_suspicion(Config.SCORE_LOITERING, "Loitering")
            else:
                person.clear_alerts("Loitering")

        # 2. Pacing Detection
        if person.role != "Staff":
            if BehaviorEngine.check_pacing(person):
                person.add_suspicion(Config.SCORE_PACING, "Pacing")
            else:
                person.clear_alerts("Pacing")

        # 3. Running/Panic Detection (High Velocity)
        # (Simple velocity threshold check)
        if person.velocity_history and np.mean(person.velocity_history) > 200: # Threshold depends on scene
            person.movement_state = "Running"
        elif person.velocity_history and np.mean(person.velocity_history) > 50:
             person.movement_state = "Walking"
        else:
             person.movement_state = "Standing"


    @staticmethod
    def check_loitering(person):
        """
        Returns True if person has stayed within a small radius for > T seconds.
        """
        if person.age_on_camera < Config.LOITERING_TIME_SECONDS:
            return False

        # Get history points from the requisite time window
        # Since history is a deque of (point, time), we look back
        recent_points = [p[0] for p in person.location_history]
        
        if len(recent_points) < 20: 
            return False

        # Calculate standard deviation of positions
        std_dev = np.std(recent_points, axis=0)
        avg_spread = np.mean(std_dev)

        # If spread is small but time is long -> Loitering
        return avg_spread < Config.LOITERING_RADIUS_PITCH

    @staticmethod
    def check_pacing(person):
        """
        Returns True if person is moving back and forth repeatedly.
        """
        if len(person.location_history) < 50:
            return False
            
        # Simplified pacing: Standard deviation in one axis is high, 
        # but displacement (start vs end) is low.
        
        points = np.array([p[0] for p in person.location_history])
        start_point = points[0]
        end_point = points[-1]
        
        displacement = np.linalg.norm(end_point - start_point)
        total_distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        if total_distance == 0: return False

        ratio = displacement / total_distance
        
        # Low ratio means lots of movement but little progress (Pacing/Circling)
        return ratio < 0.1 and total_distance > 500  # Ensure they actually moved significant amount
