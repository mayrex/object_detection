from ultralytics import YOLO
import cv2

model = YOLO("./model/detectionv3.pt")

results = model.predict(source="dataset/images/train/000_1_0001.png", show=True)

print(results)

