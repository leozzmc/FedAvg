from ultralytics import YOLO
import cv2
import csv


def count_leaf_number(model_path, img_path):
# Must load the after-train model
    model = model_path
    img = cv2.imread(img_path)
    results = model(img)
    leaf_count = sum([1 for box in results[0].boxes.data if model.names[int(box[5])] == 'leaf'])
    return leaf_count
