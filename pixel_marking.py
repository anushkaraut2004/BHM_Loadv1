from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# SET VIDEO NAME HERE (ONLY CHANGE THIS)
# =====================================================
VIDEO_NAME = "pic_2.mp4"   # change to any video inside /video folder

# =====================================================
# AUTO PATH BUILDING
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
VIDEO_FILE = BASE_DIR / "video" / VIDEO_NAME
OUT_FILE = BASE_DIR / "Pixel_image" / "frame_with_pixel_grid_random.png"

print("Using video:", VIDEO_FILE)

if not VIDEO_FILE.exists():
    raise FileNotFoundError(f"❌ {VIDEO_NAME} not found inside 'video' folder")

# =====================================================
# EXTRACT FRAME
# =====================================================
cap = cv2.VideoCapture(str(VIDEO_FILE))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = frame_count // 2  # middle frame

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("❌ Could not extract frame.")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w, _ = frame_rgb.shape

plt.figure(figsize=(12, 7))
plt.imshow(frame_rgb)

# Grid spacing
step = 200

# Vertical lines + X labels
for x in range(0, w, step):
    plt.axvline(x=x, linewidth=0.5)
    plt.text(x, 20, str(x), fontsize=8)

# Horizontal lines + Y labels
for y in range(0, h, step):
    plt.axhline(y=y, linewidth=0.5)
    plt.text(10, y, str(y), fontsize=8)

plt.title(f"{VIDEO_NAME} | Frame {frame_idx} with Pixel Grid")
plt.axis('off')

plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
plt.show()

print("Saved:", OUT_FILE)