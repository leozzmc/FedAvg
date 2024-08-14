import cv2
import os
import numpy as np

def draw_bbox_and_extract_roi(image_path, x, y, width, height, output_path):
    # 讀取圖像
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法加載圖像：{image_path}")
        return False
    else:
        print(f"已加載圖像：{image_path}")

    # 繪製邊界框
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

    # 提取ROI
    roi = img[y:y + height, x:x + width]

    # 保存處理後的全圖和ROI
    cv2.imwrite(f"{output_path}/processed_image.jpg", img)
    cv2.imwrite(f"{output_path}/roi.jpg", roi)

    return True

# 假設的位置數據
positions = {
    'x': 100,
    'y': 50,
    'width': 150,
    'height': 100
}

# 處理第一張圖片
root = os.getcwd()
image_path1 = root + '/orchid_image/100_1.jpg'  # 請替換為正確的文件路徑
output_path = root           # 請替換為希望保存文件的路徑

if draw_bbox_and_extract_roi(image_path1, **positions, output_path=output_path):
    print("圖像已處理並保存。")
else:
    print("圖像處理失敗，請檢查輸入。")
