from ultralytics import YOLO
import cv2
import os

def calculate_leaf_area(model, img_path):
    img = cv2.imread(img_path)
    results = model(img)

    areas = {}
    for idx, box in enumerate(results[0].boxes.data):
        # 假設 box[0] 和 box[1] 是邊界框的坐標
        x1, y1, x2, y2 = box[:4].int().tolist()
        width = x2 - x1
        height = y2 - y1
        area = width * height  # 計算面積
        
        # 使用 ID 和編號作為鍵
        image_id = os.path.basename(img_path).split('_')[0]  # 提取圖片名稱中的 ID
        id_number = f"{idx + 1}"  # 編號從 1 開始
        areas[f"{image_id}_{id_number}"] = area

    return areas 

# 假設距離 葉子 50 公分