import cv2
import csv
import time
import threading
import uuid
import numpy as np
from collections import deque
from ultralytics import YOLO
import easyocr
import os

# ---------------- Configuration ----------------
MODEL_PATH       = r"C:\Users\sujaysantanu akshay\Desktop\Helmet Detection_YOLOv8.v3-with-and-without-helmet-dataset.yolov8\helmet_training\helmet_exp1\weights\best.pt"
RTSP_URL         = r"rtsp://admin:admin%404321@112.133.239.23:554/cam/realmonitor?channel=1&subtype=0"
CONF_THR         = 0.5
PRE_BUFFER_SEC   = 3     # seconds before violation
POST_BUFFER_SEC  = 3     # seconds after violation
BUFFER_FPS       = 20
CSV_LOG          = "helmet_violations.csv"
BASE_DIR         = "violations"
os.makedirs(BASE_DIR, exist_ok=True)

# ---------------- Initialize Models ----------------
helmet_model = YOLO(MODEL_PATH)
helmet_model.conf = CONF_THR
plate_reader  = easyocr.Reader(['en'])

# ---------------- Prepare CSV ----------------
if not os.path.isfile(CSV_LOG):
    with open(CSV_LOG, 'a', newline='') as f:
        csv.writer(f).writerow(["id","timestamp","label","confidence","plate_text","screenshot","video"])

# ---------------- Buffers ----------------
pre_buffer = deque(maxlen=int(PRE_BUFFER_SEC * BUFFER_FPS))

# ---------------- Video Stream ----------------
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret:
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(RTSP_URL)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.stopped = True
        self.cap.release()

# ---------------- Main Setup ----------------
stream = VideoStream(RTSP_URL)
while True:
    ret, frame = stream.read()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        break
    time.sleep(0.1)

# Define trigger zone
zone_margin = int(0.12 * w)
zone_height = int(0.75 * h)
TRIGGER_ZONE = (zone_margin, h-zone_height, w-zone_margin, h-int(0.02*h))

# Display window
win = "Helmet Detector"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, w, h)
prev_time = time.time()

# -------------- Helper ----------------
def save_violation(frames_before, frames_after, frame):
    vid_id = uuid.uuid4().hex
    ts = time.strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(BASE_DIR, f"{ts}_{vid_id}")
    os.makedirs(folder, exist_ok=True)
    video_path = os.path.join(folder, f"{vid_id}.mp4")
    screenshot_path = os.path.join(folder, f"{vid_id}.jpg")
    # save screenshot
    cv2.imwrite(screenshot_path, frame)
    # save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, BUFFER_FPS, (w, h))
    for f in frames_before: out.write(f)
    for f in frames_after: out.write(f)
    out.release()
    return vid_id, ts, screenshot_path, video_path

# ---------------- Main Loop ----------------
while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(blank, "Reconnecting...", (int(w*0.3), h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow(win, blank)
        if cv2.waitKey(1000) == ord('q'):
            break
        continue

    # buffer
    pre_buffer.append(frame.copy())

    # draw zone
    x1, y1, x2, y2 = TRIGGER_ZONE
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)

    # inference & annotate
    results = helmet_model(frame)[0]
    violation = False
    viol_conf = 0
    for bbox, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = helmet_model.names[int(cls)].lower()
        bx1, by1, bx2, by2 = map(int, bbox)
        cx, cy = (bx1+bx2)//2, (by1+by2)//2
        color = (0,255,0) if 'helmet' in label else (0,0,255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if 'helmet' not in label and x1<cx<x2 and y1<cy<y2:
            violation = True
            viol_conf = conf

    if violation:
        # collect post frames
        post_buffer = []
        for _ in range(int(POST_BUFFER_SEC * BUFFER_FPS)):
            r2, f2 = stream.read()
            if not r2: break
            post_buffer.append(f2.copy())
        # save
        vid_id, ts, scr_path, vid_path = save_violation(list(pre_buffer), post_buffer, frame)
        # OCR
        ocr = plate_reader.readtext(frame)
        plates = [t[1] for t in ocr if t[2] > 0.5]
        plate = plates[0] if plates else 'UNKNOWN'
        # log
        with open(CSV_LOG, 'a', newline='') as f:
            csv.writer(f).writerow([vid_id, ts, 'no_helmet', f"{viol_conf:.2f}", plate, scr_path, vid_path])
        pre_buffer.clear()

    # FPS
    now = time.time()
    fps = 1/(now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow(win, frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()