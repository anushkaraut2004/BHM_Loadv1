import cv2
import easyocr
import re

# -----------------------------
# LOAD OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture("video/pic_2.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_id = 0

base_time = None

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Smaller ROI to remove "Device"
    roi = frame[5:45, 0:520]

    # OCR once per second
    if frame_id % fps == 0:

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

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

        # -----------------------------
        # CLEAN TEXT
        # -----------------------------
        text_all = text_all.replace(".", "/")
        text_all = text_all.replace(" ", "")

        # Extract timestamp
        match = re.search(
            r'(\d{4})/(\d{2})/(\d{2})(\d{2}):(\d{2})',
            text_all
        )

        if match:

            year  = match.group(1)
            month = match.group(2)
            day   = match.group(3)
            hour  = match.group(4)
            minute= match.group(5)

            # Fix wrong OCR years
            if year.startswith("22") or year.startswith("28"):
                year = "2026"

            base_time = f"{year}/{month}/{day} {hour}:{minute}"

    # -----------------------------
    # GENERATE SECONDS
    # -----------------------------
    seconds = frame_id // fps % 60

    timestamp = None

    if base_time:
        timestamp = f"{base_time}:{seconds:02d}"

    print("Detected:", timestamp)

    # -----------------------------
    # DRAW ROI
    # -----------------------------
    cv2.rectangle(frame,(0,5),(520,45),(0,255,255),2)

    cv2.imshow("frame", frame)
    cv2.imshow("roi", roi)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()