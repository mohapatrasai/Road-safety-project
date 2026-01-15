from ultralytics import YOLO

# ✅ Load the correct trained model path
model = YOLO("C:/Users/deepa/OneDrive/Desktop/yolov5/runs/train/road_safety_model/weights/best.pt")

# ✅ Provide your test image folder
results = model.predict(source="C:/Users/deepa/OneDrive/Desktop/test_images", save=True, conf=0.25)

print("✅ Prediction complete. Check yolov5/runs/detect/predict/ for results.")

