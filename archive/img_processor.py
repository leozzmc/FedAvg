import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 提取 ROI 和繪製邊框
def extract_roi(arr, x, y, w, h, intensity, line):
    roi = arr[y:y+h, x:x+w].copy()

    bounding_box = arr.copy()
    cv2.rectangle(bounding_box, (x, y), (x+w, y+h), intensity, line)

    return (roi, bounding_box)

# 確保 output 資料夾存在
output_dir = 'train'
roi_output_dir = os.path.join(output_dir, 'roi')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(roi_output_dir):
    os.makedirs(roi_output_dir)

# 使用一般圖像
root = os.getcwd()
Pth = root + "/orchid_image/"  # 圖片路徑
filenames = [f for f in os.listdir(Pth) if os.path.isfile(os.path.join(Pth, f))]

df = pd.read_excel('./dataset2.xlsx', sheet_name='train')

for filename in filenames:

    image_path = os.path.join(Pth, filename)

    # 檢查圖像副檔名
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension != '.jpg':
        # 將非 .jpg 圖片讀取並保存為 .jpg
        image = cv2.imread(image_path)
        if image is not None:
            # 將圖像保存為 .jpg
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            new_image_path = os.path.join(Pth, new_filename)
            cv2.imwrite(new_image_path, image)
            print(f"Converted and saved {filename} to {new_filename}")
            
            # 刪除原始的非 .jpg 圖片檔案
            os.remove(image_path)
            print(f"Deleted original file: {filename}")

            # 使用 .jpg 文件名繼續後續處理
            filename = new_filename
            image_path = new_image_path

    # 使用 OpenCV 讀取圖像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 去掉檔案的副檔名 (例如 .jpg)
    filename_without_extension = os.path.splitext(filename)[0]

    # 從 Excel 檔案中讀取位置資訊
    pos = df[df['filename'] == filename_without_extension]

    if not pos.empty:
        # 取得對應的標籤名稱
        label = pos['label'].values[0]
        pos = pos[['left_x', 'top_y', 'width', 'height']].values.flatten().tolist()
        ori_x = pos[0]
        ori_y = pos[1]
        leaf_width = pos[2]
        leaf_height = pos[3]

        # 設定邊框和 ROI
        intensity = (255, 0, 0)  # 用於邊框的紅色
        line = 2  # 邊框的寬度

        # 擷取 ROI 並繪製邊框
        roi, bounding_boxed = extract_roi(image, ori_x, ori_y, leaf_width, leaf_height, intensity, line)

        # 顯示並保存結果圖片
        output_path = os.path.join(output_dir, f'output_image_{label}.jpg')
        plt.imshow(bounding_boxed)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

        # 保存 ROI 圖片
        roi_path = os.path.join(roi_output_dir, f'roi_{label}.png')
        plt.imsave(roi_path, roi)
    else:
        print(f"No matching data found for {filename_without_extension}")
        # 刪除沒有匹配數據的圖像
        os.remove(image_path)
        print(f"Deleted image with no matching data: {filename}")
