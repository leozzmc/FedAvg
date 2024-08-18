import pandas as pd
import os
import cv2

def convert_to_yolo_format(csv_path, images_folder, output_folder):
    # 讀取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 確保輸出目錄存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 讀取所有圖像檔案
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    
    if not image_files:
        print(f"No images found in {images_folder}.")
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image {image_file} could not be read.")
            continue
        
        height, width, _ = image.shape
        
        # 去掉檔案的副檔名
        base_name = os.path.splitext(image_file)[0]
        
        # 獲取對應的標註
        annotations = df[df['filename'] == base_name]
        
        if annotations.empty:
            print(f"No annotations found for {base_name}.")
            continue
        
        # 為每個圖像生成單獨的標註文件
        output_annotation_file = os.path.join(output_folder, f"{base_name}.txt")
        with open(output_annotation_file, 'w') as f:
            for _, row in annotations.iterrows():
                # class_id = row['label']
                class_id = 0
                x = row['left_x']
                y = row['top_y']
                w = row['width']
                h = row['height']
                
                # 計算 YOLO 格式的值
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_ratio = w / width
                h_ratio = h / height
                
                f.write(f"{class_id} {x_center} {y_center} {w_ratio} {h_ratio}\n")

# 設定檔案路徑
csv_path = './leaf_location.csv'  # CSV 文件的路徑
images_folder = 'orchid_image/'    # 圖像資料夾
output_folder = 'labels/'  # YOLO 格式的標註資料夾

convert_to_yolo_format(csv_path, images_folder, output_folder)
