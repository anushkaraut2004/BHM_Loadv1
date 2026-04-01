#!/usr/bin/env python3
import cv2
import csv
import re
from datetime import datetime, timedelta
import easyocr
from ultralytics import YOLO
from collections import defaultdict, deque

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "video/"
OUTPUT_VIDEO = "output_detection.mp4"
OUTPUT_CSV = "vehicle_crossings.csv"

YOLO_MODEL = "yolo11m.pt"
TRACKER_CFG = "bytetrack.yaml"

# detect vehicles
VEHICLE_CLASSES = {2,3,5,7}

LINE_P1 = (986,633)
LINE_P2 = (1467,534)

TIMESTAMP_ROI = (0,0,1200,70)

# refined thresholds
TEMPO_THRESHOLD = 0.09
LIGHT_TRUCK_THRESHOLD = 0.16
MOTORCYCLE_THRESHOLD = 0.035

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def side_of_line(px,py,x1,y1,x2,y2):
    return (px-x1)*(y2-y1)-(py-y1)*(x2-x1)

def parse_ocr_to_datetime(text):

    txt = text.replace(".", "-").replace("/", "-")
    txt = txt.replace(" ","")

    m = re.search(
        r'(\d{4})-(\d{2})-(\d{2})(\d{2}):(\d{2}):(\d{2})(AM|PM)?',
        txt,
        flags=re.IGNORECASE
    )

    if not m:
        return None

    year,mon,day,hh,mm,ss,ampm = m.groups()

    y,mo,d = int(year),int(mon),int(day)
    h,mi,s = int(hh),int(mm),int(ss)

    if ampm:
        ampm = ampm.upper()

        if ampm=="PM" and h!=12:
            h+=12

        if ampm=="AM" and h==12:
            h=0

    try:
        return datetime(y,mo,d,h,mi,s)
    except:
        return None


# -----------------------------
# LOAD MODELS
# -----------------------------
print("\n🚀 Loading YOLO11...")
model = YOLO(YOLO_MODEL)

print("🔎 Loading EasyOCR...")
reader = easyocr.Reader(['en'],gpu=False)

# -----------------------------
# VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
fps = float(fps)

width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width,height)
)

# -----------------------------
# CSV
# -----------------------------
csv_file = open(OUTPUT_CSV,"w",newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "vehicle",
    "track_id",
    "frame"
])

# -----------------------------
# MEMORY
# -----------------------------
track_history = defaultdict(lambda:deque(maxlen=5))
track_lengths = defaultdict(list)

counted=set()

last_ocr_time=None
last_ocr_frame=None

frame_id=0

print("\n📹 Processing started...\n")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret,frame=cap.read()
    if not ret:
        break

    frame_id+=1

# ---------------- OCR ----------------

    if frame_id % int(fps)==0:

        x1,y1,x2,y2=TIMESTAMP_ROI
        roi=frame[y1:y2,x1:x2]

        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,None,fx=3,fy=3)
        gray=cv2.GaussianBlur(gray,(3,3),0)

        _,thr=cv2.threshold(
            gray,0,255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )

        res=reader.readtext(thr)

        txt=" ".join([t for (_,t,_) in res])

        parsed=parse_ocr_to_datetime(txt)

        if parsed:
            last_ocr_time=parsed
            last_ocr_frame=frame_id

# ---------------- YOLO TRACKING ----------------

    results=model.track(
        frame,
        persist=True,
        tracker=TRACKER_CFG,
        conf=0.20,        # better motorcycle detection
        iou=0.5,
        verbose=False
    )

    r=results[0]

    if r.boxes is not None and r.boxes.id is not None:

        boxes=r.boxes.xyxy
        ids=r.boxes.id
        classes=r.boxes.cls

        for box,tid,cls in zip(boxes,ids,classes):

            cls=int(cls)

            if cls not in VEHICLE_CLASSES:
                continue

            tid=int(tid)

            x1,y1,x2,y2=map(int,box)

            cx=(x1+x2)//2
            cy=(y1+y2)//2

            track_history[tid].append((cx,cy))

# ---------------- VEHICLE SIZE ----------------

            bbox_width = x2-x1
            frame_width = frame.shape[1]

            ratio = bbox_width/frame_width

            track_lengths[tid].append(ratio)
            avg_ratio = sum(track_lengths[tid]) / len(track_lengths[tid])

# ---------------- VEHICLE CLASSIFICATION ----------------

            label=model.names[cls]

            if label=="motorcycle":
                vehicle="motorcycle"

            elif label=="car":

                if avg_ratio < MOTORCYCLE_THRESHOLD:
                    vehicle="motorcycle"
                else:
                    vehicle="car"

            elif label=="bus":
                vehicle="bus"

            elif label=="truck":

                if avg_ratio < TEMPO_THRESHOLD:
                    vehicle="tempo"

                elif avg_ratio < LIGHT_TRUCK_THRESHOLD:
                    vehicle="light_truck"

                else:
                    vehicle="truck"

            else:
                vehicle=label

# ---------------- LINE CROSSING ----------------

            if len(track_history[tid])>=2:

                prev=track_history[tid][-2]

                prev_side=side_of_line(prev[0],prev[1],*LINE_P1,*LINE_P2)
                curr_side=side_of_line(cx,cy,*LINE_P1,*LINE_P2)

                if prev_side<0 and curr_side>=0:

                    if tid not in counted:

                        counted.add(tid)

                        if last_ocr_time:

                            delta=(frame_id-last_ocr_frame)/fps
                            cross_time=last_ocr_time+timedelta(seconds=delta)

                        else:
                            cross_time=datetime.now()

                        timestamp=cross_time.strftime("%Y/%m/%d %H:%M:%S")

                        csv_writer.writerow([
                            timestamp,
                            vehicle,
                            tid,
                            frame_id
                        ])

                        csv_file.flush()

                        print(
                            f"🚗 CROSSING | {vehicle.upper():12} | ID {tid:3} | {timestamp}"
                        )

# ---------------- DRAW ----------------

            color=(0,0,255) if tid in counted else (0,255,0)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,
                f"{vehicle} {tid}",
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.circle(frame,(cx,cy),3,(255,0,0),-1)

# ---------------- DRAW LINE ----------------

    cv2.line(frame,LINE_P1,LINE_P2,(255,0,0),3)

# ---------------- DISPLAY TIME ----------------

    if last_ocr_time:

        delta=(frame_id-last_ocr_frame)/fps
        current_time=last_ocr_time+timedelta(seconds=delta)

        display_time=current_time.strftime("%Y/%m/%d %H:%M:%S")

        cv2.putText(
            frame,
            display_time,
            (20,95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

# ---------------- SAVE VIDEO ----------------

    out.write(frame)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1)==27:
        break

# ---------------- CLEANUP ----------------

cap.release()
out.release()
csv_file.close()

cv2.destroyAllWindows()

print("\n✅ Processing finished")