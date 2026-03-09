import cv2

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture("video/pic.mp4")

# -----------------------------
# MOUSE FUNCTION
# -----------------------------
def mouse_position(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        print("X:", x, "Y:", y)

# -----------------------------
# READ VIDEO
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("frame", frame)

    # attach mouse callback
    cv2.setMouseCallback("frame", mouse_position)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()