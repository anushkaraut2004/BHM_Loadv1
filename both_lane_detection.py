#!/usr/bin/env python3
import cv2
import csv
import re
from datetime import datetime, timedelta
import numpy as np
import easyocr
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "video/pic.mp4"          # input video
OUTPUT_VIDEO = "output_detection.mp4" # output annotated video
OUTPUT_CSV = "vehicle_crossings.csv"  # output CSV

YOLO_MODEL_PATH = "yolov8n.pt"        # YOLO model path
TRACKER_CFG = "bytetrack.yaml"        # ByteTrack config
USE_GPU_OCR = False                   # set True if EasyOCR with GPU is desired

# vehicle COCO class ids we want (common vehicle classes)
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# line endpoints (given)
LINE_P1 = (986, 633)
LINE_P2 = (1467, 534)

# OCR ROI (same as your code)
TIMESTAMP_ROI = (0, 0, 1200, 70)  # x1,y1,x2,y2

# OCR frequency: run OCR once per second (will compute from video FPS)
OCR_INTERVAL_SECONDS = 1

# -----------------------------
# HELPERS
# -----------------------------
def side_of_line(px, py, x1, y1, x2, y2):
    """Signed cross product — positive/negative side of directed line (x1,y1)->(x2,y2)."""
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

def sign_val(v):
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0

def proj_param_on_segment(px, py, x1, y1, x2, y2):
    """Return projection parameter t for point projection onto line segment.
       t in [0,1] means projection lies on segment."""
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    denom = vx*vx + vy*vy
    if denom == 0:
        return None
    t = (wx*vx + wy*vy) / denom
    return t

def parse_ocr_to_datetime(text):
    """Parse OCR text using the same regex you used and convert to a Python datetime in 24-hour."""
    # normalize separators
    txt = text.replace(".", "-").replace("/", "-")
    txt = txt.replace(" ", "")
    # regex similar to your original (accepts AM/PM optionally)
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})(\d{2}):(\d{2}):(\d{2})(AM|PM)?', txt, flags=re.IGNORECASE)
    if not m:
        return None
    year, mon, day, hh, mm, ss, ampm = m.groups()
    y, mo, d = int(year), int(mon), int(day)
    h, mi, s = int(hh), int(mm), int(ss)
    if ampm:
        ampm = ampm.upper()
        if ampm == "PM" and h != 12:
            h += 12
        if ampm == "AM" and h == 12:
            h = 0
    try:
        return datetime(y, mo, d, h, mi, s)
    except Exception:
        return None

# -----------------------------
# INIT MODELS + I/O
# -----------------------------
print("Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)
print("YOLO loaded.")

print("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=USE_GPU_OCR)
print("OCR loaded.")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fps = float(fps)
ocr_interval_frames = max(1, int(round(OCR_INTERVAL_SECONDS * fps)))

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

csv_f = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_f)
csv_writer.writerow(["timestamp", "vehicle_type", "track_id", "frame_index"])

# line endpoints
x1_line, y1_line = LINE_P1
x2_line, y2_line = LINE_P2

# decide which side is "below" (sample a point slightly below midpoint)
mid_x = (x1_line + x2_line) / 2.0
mid_y = (y1_line + y2_line) / 2.0
sample_below = side_of_line(mid_x, mid_y + 50, x1_line, y1_line, x2_line, y2_line)
below_sign = sign_val(sample_below)
if below_sign == 0:
    below_sign = 1  # fallback

print(f"Line: {LINE_P1} -> {LINE_P2}, below_sign={below_sign}")

# tracking memory
previous_side = {}   # track_id -> previous side value
crossed_ids = set()  # track ids that have been counted

last_ocr_dt = None
last_ocr_frameidx = None
video_start_dt = datetime.now()  # fallback base time if OCR never succeeds

frame_idx = 0
print("Starting processing (press ESC to stop)...")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # -----------------------------
    # OCR (once per second approx)
    # -----------------------------
    if frame_idx % ocr_interval_frames == 0:
        x1_roi, y1_roi, x2_roi, y2_roi = TIMESTAMP_ROI
        roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            ocr_results = reader.readtext(thresh)
            all_text = " ".join([t for (_, t, _) in ocr_results])
        except Exception as e:
            all_text = ""
        parsed_dt = parse_ocr_to_datetime(all_text)
        if parsed_dt:
            last_ocr_dt = parsed_dt
            last_ocr_frameidx = frame_idx
            print(f"[OCR] frame={frame_idx} parsed={last_ocr_dt} raw='{all_text}'")
        else:
            # optional debug: print failure occasionally
            # print(f"[OCR FAIL] frame={frame_idx} raw='{all_text}'")
            pass

    # -----------------------------
    # YOLO tracking (ByteTrack)
    # -----------------------------
    results = model.track(frame, persist=True, tracker=TRACKER_CFG, conf=0.35, iou=0.5, verbose=False)
    r = results[0]

    if r.boxes is not None and getattr(r.boxes, "id", None) is not None:
        boxes = r.boxes.xyxy
        ids = r.boxes.id
        classes = r.boxes.cls

        for box, tid, cls in zip(boxes, ids, classes):
            cls = int(cls)
            if cls not in VEHICLE_CLASSES:
                continue

            tid = int(tid)
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # compute side and projection param
            curr_side = side_of_line(cx, cy, x1_line, y1_line, x2_line, y2_line)
            prev_side = previous_side.get(tid, None)
            t_proj = proj_param_on_segment(cx, cy, x1_line, y1_line, x2_line, y2_line)
            on_segment = (t_proj is not None and 0.0 <= t_proj <= 1.0)

            crossed_now = False
            # below -> above : previous side had same sign as below_sign, current side opposite
            if prev_side is not None:
                if sign_val(prev_side) == below_sign and sign_val(curr_side) == -below_sign and on_segment:
                    if tid not in crossed_ids:
                        crossed_now = True
                        crossed_ids.add(tid)

                        # compute exact timestamp for crossing frame:
                        if last_ocr_dt is not None and last_ocr_frameidx is not None:
                            delta_frames = frame_idx - last_ocr_frameidx
                            crossing_dt = last_ocr_dt + timedelta(seconds=(delta_frames / fps))
                        else:
                            crossing_dt = video_start_dt + timedelta(seconds=(frame_idx / fps))

                        timestamp_str = crossing_dt.strftime("%Y/%m/%d %H:%M:%S")  # 24-hour

                        label = model.names[cls]
                        csv_writer.writerow([timestamp_str, label, tid, frame_idx])
                        csv_f.flush()
                        print(f"[CROSS] id={tid} type={label} frame={frame_idx} time={timestamp_str}")

            previous_side[tid] = curr_side

            # draw bounding box (green before crossing, red after)
            color = (0, 255, 0) if tid not in crossed_ids else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{model.names[cls]} ID:{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    # -----------------------------
    # draw ROI and line on frame and show last OCR (optional)
    # -----------------------------
    x1_roi, y1_roi, x2_roi, y2_roi = TIMESTAMP_ROI
    cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 255), 1)
    cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), 3)
    if last_ocr_dt:
        cv2.putText(frame, last_ocr_dt.strftime("%Y/%m/%d %H:%M:%S"),
                    (10, y2_roi + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # write frame to output video
    out.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
out.release()
csv_f.close()
cv2.destroyAllWindows()
print("Done. CSV ->", OUTPUT_CSV, " Video ->", OUTPUT_VIDEO)