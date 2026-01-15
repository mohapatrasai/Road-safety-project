# train.py
from ultralytics import YOLO

# ── STEP 1: Base model ───────────────────────────────────────────────────
# If you already have a trained checkpoint at C:\...\, put that full path here (check spelling):
# base_model = r"C:\Users\sujaysantanu akshay\Desktop\Traffic Test\hemletYoloV8_100epochs.pt"

# Otherwise, to train from scratch, just use any official YOLOv8 weight:
base_model = "yolov8n.pt"

# ── STEP 2: Path to data.yaml ─────────────────────────────────────────────
# (be sure data.yaml is in the same folder as this train.py)
data_yaml = r"C:\Users\sujaysantanu akshay\Desktop\Helmet Detection_YOLOv8.v3-with-and-without-helmet-dataset.yolov8\data.yaml"

# ── STEP 3: Instantiate model ────────────────────────────────────────────
model = YOLO(base_model)

# ── STEP 4: Launch training ───────────────────────────────────────────────
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    project="helmet_training",
    name="helmet_exp1",
    exist_ok=True,
)