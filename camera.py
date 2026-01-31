# camera.py
import cv2
import os
from face_ai import load_suspect, find_match

# ===============================
# Load suspect image (absolute path)
# ===============================
suspect_image_path = os.path.join(os.getcwd(), "data", "suspects", "ali.jpg")
load_suspect("ALI", suspect_image_path)

# ===============================
# Start webcam
# ===============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===============================
    # Find match in current frame
    # ===============================
    results = find_match(frame)

    # ===============================
    # Draw boxes & labels
    # ===============================
    for box, name, score in results:
        x1, y1, x2, y2 = box
        color = (0,0,255) if name != "Unknown" else (0,255,0)
        label = f"{name} ({score:.2f})" if name != "Unknown" else "Normal"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ===============================
    # Show frame
    # ===============================
    cv2.imshow("Smart Airport CCTV", frame)
    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# Release resources
# ===============================
cap.release()
cv2.destroyAllWindows()
