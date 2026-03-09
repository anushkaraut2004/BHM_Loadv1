from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime, timedelta
import numpy as np

# ================================
# FILE PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "video", "f00494_clean.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

OUTPUT_VIDEO = os.path.join(BASE_DIR, "line_cross_output.mp4")
OUTPUT_CSV = os.path.join(BASE_DIR, "vehicle_crossings.csv")

# ================================
# LINE COORDINATES (YOU EDIT)
# ================================
LINE_START = (1150, 1350)
LINE_END   = (1800, 1450)

# ================================
# VEHICLE CLASSES
# ================================
ALLOWED_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# ================================
# LOAD MODEL
# ================================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ================================
# OPEN VIDEO
# ================================
cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ================================
# CSV FILE
# ================================
csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "track_id",
    "vehicle_type",
    "crossing_timestamp"
])

# ================================
# MEMORY STORAGE
# ================================
previous_side = {}
crossed_ids = set()

VIDEO_START_TIME = datetime.now()
frame_idx = 0

# ================================
# FUNCTION: POINT SIDE OF LINE
# ================================
def get_side(px, py, x1, y1, x2, y2):
    return (px - x1)*(y2 - y1) - (py - y1)*(x2 - x1)

print("Starting detection...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    r = results[0]

    # draw counting line
    cv2.line(frame, LINE_START, LINE_END, (0,0,255), 3)

    timestamp = (
        VIDEO_START_TIME +
        timedelta(seconds = frame_idx / fps)
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    if r.boxes is not None and r.boxes.id is not None:

        for box, tid in zip(r.boxes, r.boxes.id):

            cls = model.names[int(box.cls[0])]
            if cls not in ALLOWED_CLASSES:
                continue

            tid = int(tid)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            # draw box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)

            side = get_side(
                cx,cy,
                LINE_START[0],LINE_START[1],
                LINE_END[0],LINE_END[1]
            )

            if tid not in previous_side:
                previous_side[tid] = side
                continue

            # check crossing
            if previous_side[tid] * side < 0:

                if tid not in crossed_ids:

                    crossed_ids.add(tid)

                    csv_writer.writerow([
                        tid,
                        cls,
                        timestamp
                    ])

                    print(
                        f"Vehicle crossed: ID={tid}, "
                        f"type={cls}, time={timestamp}"
                    )

            previous_side[tid] = side

    out.write(frame)

    frame_idx += 1

# ================================
# CLEANUP
# ================================
cap.release()
out.release()
csv_file.close()

print("Finished")