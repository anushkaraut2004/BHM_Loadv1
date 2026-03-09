#!/usr/bin/env python3
"""
Extract a frame at a user-specified timestamp from a video (in video/ folder),
draw a pixel grid with coordinates and save the image to Pixel_images/.
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def parse_time(ts: str) -> float:
    """
    Parse timestamp string and return seconds as float.
    Accepts:
      "12.34"            -> seconds
      "mm:ss"            -> minutes:seconds
      "hh:mm:ss"         -> hours:minutes:seconds
      "hh:mm:ss.msec"    -> with milliseconds
      "mm:ss.msec"
    """
    if ts is None:
        raise ValueError("No timestamp provided")
    ts = ts.strip()
    if ts.count(":") == 0:
        # simple seconds (possibly float)
        return float(ts)
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    # support mm:ss, hh:mm:ss
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60.0 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    else:
        raise ValueError("Unsupported timestamp format")

def main():
    parser = argparse.ArgumentParser(description="Extract frame at timestamp and overlay pixel grid.")
    parser.add_argument("--video-name", required=True, help="Video filename inside video/ (e.g. 4.mp4)")
    parser.add_argument("--timestamp", required=True, help="Timestamp (seconds or mm:ss or hh:mm:ss(.ms))")
    parser.add_argument("--step", type=int, default=200, help="Grid spacing in pixels (default 200)")
    parser.add_argument("--output-name", default="frame_with_pixel_grid_time.png", help="Output image name (saved in Pixel_images/)")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    VIDEO_FILE = BASE_DIR / "video" / args.video_name

    if not VIDEO_FILE.exists():
        print(f"❌ Video not found: {VIDEO_FILE}", file=sys.stderr)
        sys.exit(1)

    try:
        ts_seconds = parse_time(args.timestamp)
    except Exception as e:
        print(f"❌ Could not parse timestamp '{args.timestamp}': {e}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_FILE))
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_FILE}", file=sys.stderr)
        sys.exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
        print("⚠ Warning: FPS not detected. Defaulting to 30.0")

    video_duration = frame_count / fps if frame_count > 0 else 0.0
    if ts_seconds < 0:
        print("❌ Timestamp must be non-negative", file=sys.stderr)
        sys.exit(1)

    if ts_seconds > video_duration:
        print(f"⚠ Requested timestamp {ts_seconds:.3f}s is beyond video duration {video_duration:.3f}s. Clamping to last frame.")
        frame_idx = max(0, frame_count - 1)
    else:
        frame_idx = round(ts_seconds * fps)
        if frame_idx >= frame_count:
            frame_idx = frame_count - 1

    print(f"Using video: {VIDEO_FILE}")
    print(f"Video FPS: {fps:.3f}, Total frames: {frame_count}, Duration: {video_duration:.3f}s")
    print(f"Extracting frame at {ts_seconds:.3f}s -> frame index {frame_idx}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Failed to read frame.", file=sys.stderr)
        sys.exit(1)

    # Convert to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape

    # Prepare output folder
    OUT_FOLDER = BASE_DIR / "Pixel_images"
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUT_PATH = OUT_FOLDER / args.output_name

    # Draw grid & labels with matplotlib
    plt.figure(figsize=(12, 7))
    plt.imshow(frame_rgb)
    step = max(1, args.step)

    # Vertical lines (X coordinates)
    for x in range(0, w, step):
        plt.axvline(x=x, linewidth=0.5)
        plt.text(x + 2, 12, str(x), fontsize=8, backgroundcolor="white")

    # Horizontal lines (Y coordinates)
    for y in range(0, h, step):
        plt.axhline(y=y, linewidth=0.5)
        plt.text(6, y + 2, str(y), fontsize=8, backgroundcolor="white")

    plt.title(f"{args.video_name} | t={ts_seconds:.3f}s | frame {frame_idx}")
    plt.axis('off')

    plt.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()


# python3 timewise_pixel.py --video-name 20260224_121859_tp00451.mp4 --timestamp 5:12


# ffmpeg -i video/line_cross_output.mp4 -c:v libx264 -preset fast -crf 18 video/output_clean.mp4