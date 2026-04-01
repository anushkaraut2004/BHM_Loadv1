import cv2
import easyocr
import re
from ultralytics import YOLO

# -----------------------------
# LOAD MODELS
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

model = YOLO("yolov8n.pt")  # lightweight model

# COCO vehicle classes
vehicle_classes = [2, 3, 5, 7]  
# car, motorcycle, bus, truck

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture("video/pic.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_id = 0

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # -----------------------------
    # TIMESTAMP REGION
    # -----------------------------
    timestamp_roi = frame[0:70, 0:1200]

    if frame_id % fps == 0:

        gray = cv2.cvtColor(timestamp_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        results = reader.readtext(thresh)

        text_all = ""
        for (_, text, _) in results:
            text_all += " " + text

        text_all = text_all.replace(" ", "")

        match = re.search(
            r'(\d{4})-(\d{2})-(\d{2})(\d{2}):(\d{2}):(\d{2})(AM|PM)',
            text_all
        )

        if match:

            year   = match.group(1)
            month  = match.group(2)
            day    = match.group(3)
            hour   = match.group(4)
            minute = match.group(5)
            second = match.group(6)
            ampm   = match.group(7)

            timestamp = f"{year}/{month}/{day} {hour}:{minute}:{second} {ampm}"

            print("Timestamp:", timestamp)

    # -----------------------------
    # BRIDGE ROI (vehicle detection)
    # -----------------------------
    bridge_roi = frame[350:850, 500:1600]

    results = model(bridge_roi)

    for r in results:

        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for box, cls in zip(boxes, classes):

            cls = int(cls)

            if cls in vehicle_classes:

                x1, y1, x2, y2 = map(int, box)

                # adjust coordinates to full frame
                x1 += 500
                x2 += 500
                y1 += 350
                y2 += 350

                label = model.names[cls]

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    # -----------------------------
    # DRAW REGIONS
    # -----------------------------
    cv2.rectangle(frame,(0,0),(1200,70),(0,255,255),2)       # timestamp
    cv2.rectangle(frame,(500,350),(1600,850),(255,0,0),2)    # bridge ROI

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()