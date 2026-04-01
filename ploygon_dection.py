import cv2
import easyocr
import re
import numpy as np
from ultralytics import YOLO

# -----------------------------
# LOAD MODELS
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

model = YOLO("yolov8n.pt")

# vehicle classes from COCO
vehicle_classes = [2,3,5,7]  
# car, motorcycle, bus, truck

# -----------------------------
# BRIDGE POLYGON
# -----------------------------
bridge_polygon = np.array([
    [1107,636],
    [1083,400],
    [1315,402],
    [1463,582]
], np.int32)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture("video/pic.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_id = 0

timestamp = None

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
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        results = reader.readtext(thresh)

        text_all = ""

        for (_, text, _) in results:
            text_all += " " + text

        print("OCR RAW:", text_all)

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
    # CREATE BRIDGE MASK
    # -----------------------------
    mask = np.zeros_like(frame)

    cv2.fillPoly(mask, [bridge_polygon], (255,255,255))

    bridge_roi = cv2.bitwise_and(frame, mask)

    # -----------------------------
    # YOLO VEHICLE DETECTION
    # -----------------------------
    results = model(bridge_roi, conf=0.3)

    for r in results:

        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for box, cls in zip(boxes, classes):

            cls = int(cls)

            if cls in vehicle_classes:

                x1,y1,x2,y2 = map(int, box)

                label = model.names[cls]

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

                if timestamp:
                    cv2.putText(
                        frame,
                        timestamp,
                        (x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2
                    )

    # -----------------------------
    # DRAW REGIONS
    # -----------------------------
    cv2.rectangle(frame,(0,0),(1200,70),(0,255,255),2)

    cv2.polylines(
        frame,
        [bridge_polygon],
        True,
        (255,0,0),
        3
    )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()