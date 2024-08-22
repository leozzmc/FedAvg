from ultralytics import YOLO
import cv2


model = YOLO("Orchid_label.v5i.yolov8/runs/detect/train/weights/best.pt") 

img_path = "/Users/kuangsin/FedAvg/image_processing/client1/Orchid_label.v5i.yolov8/valid/images/32_1_JPG.rf.6d34cda5b873f30dafb865ddca2d2bef.jpg"
img = cv2.imread(img_path)


results = model(img)


leaf_count = sum([1 for box in results[0].boxes.data if model.names[int(box[5])] == 'leaf'])

print(f"Orchid leaves count: {leaf_count}")
