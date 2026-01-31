import threading
from playsound import playsound
import os
import time

from logger import system_logger

last_played_time = 0
COOLDOWN_SECONDS = 2 # Reduced for more urgent feeling

def play_sound_thread():
    global last_played_time
    try:
        sound_path = 'alert_sound.mp3'
        if os.path.exists(sound_path):
            system_logger.logger.info(f"Triggering audio alert: {sound_path}")
            playsound(sound_path)
        else:
            system_logger.logger.error(f"Alert sound missing at {os.path.abspath(sound_path)}")
    except Exception as e:
        system_logger.logger.error(f"Audio Playback Error: {e}")

def red_alert_sound():
    global last_played_time
    if time.time() - last_played_time < COOLDOWN_SECONDS:
        return 
        
    last_played_time = time.time()
    threading.Thread(target=play_sound_thread, daemon=True).start()


