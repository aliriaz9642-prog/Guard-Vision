from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame)[0]
    dangerous = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["knife", "gun", "fire"]:
            dangerous = True

    return dangerous, results
