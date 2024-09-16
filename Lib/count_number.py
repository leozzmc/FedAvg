from ultralytics import YOLO
import cv2

# Must load the after-train model
model = YOLO("") 

# Must load the client images
img_path = ""
img = cv2.imread(img_path)

results = model(img)


leaf_count = sum([1 for box in results[0].boxes.data if model.names[int(box[5])] == 'leaf'])

print(f"Orchid leaves count: {leaf_count}")
