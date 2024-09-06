import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_roi(arr, x, y, w, h):
    roi = arr[y:y+h, x:x+w].copy()
    return roi

# Make sure that 'output' dir exists
output_dir = 'train'
roi_output_dir = os.path.join(output_dir, 'roi')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(roi_output_dir):
    os.makedirs(roi_output_dir)

root = os.getcwd()
Pth = root + "/orchid_image/"  # Source image path
filenames = [f for f in os.listdir(Pth) if os.path.isfile(os.path.join(Pth, f))]

df = pd.read_excel('./dataset2.xlsx', sheet_name='train')

for filename in filenames:
    image_path = os.path.join(Pth, filename)

    # check file extension
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension != '.jpg':
        # non-jpg file will save to .jpg file
        image = cv2.imread(image_path)
        if image is not None:
            # save to .jpg
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            new_image_path = os.path.join(Pth, new_filename)
            cv2.imwrite(new_image_path, image)
            print(f"Converted and saved {filename} to {new_filename}")
            
            # Delete original non-jpg images
            os.remove(image_path)
            print(f"Deleted original file: {filename}")

            filename = new_filename
            image_path = new_image_path

    # Use OpenCV to read images
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 去掉檔案的副檔名 (例如 .jpg)
    filename_without_extension = os.path.splitext(filename)[0]

    # Extract leaf position from the excel file
    pos = df[df['filename'] == filename_without_extension]

    if not pos.empty:
        label = pos['label'].values[0]
        pos = pos[['left_x', 'top_y', 'width', 'height']].values.flatten().tolist()
        ori_x = pos[0]
        ori_y = pos[1]
        leaf_width = pos[2]
        leaf_height = pos[3]

        # 設定裁切參數
        target_len = 40  # 裁切區域大小
        interval = 20    # 裁切間隔
        coords = []
        cur_x = ori_x
        cur_y = ori_y
        x_limit = ori_x + leaf_width
        y_limit = ori_y + leaf_height
        while (cur_x + target_len <= x_limit):
            while (cur_y + target_len <= y_limit):
                coords.append((cur_x, cur_y))
                cur_y += interval
            cur_x += interval
            cur_y = ori_y

        # 裁切並保存 ROI 圖片
        cnt = 1
        for coordinate in coords:
            (x, y) = coordinate
            roi = extract_roi(image, x, y, target_len, target_len)
            if roi.size == 0:
                continue
            if np.isnan(roi).any():
                continue
            if np.isinf(roi).any():
                continue
            roi_path = os.path.join(roi_output_dir, f'roi_{label}_{cnt}.png')
            plt.imsave(roi_path, roi)
            cnt += 1
    else:
        print(f"No matching data found for {filename_without_extension}")
        # Delete unmatched images
        os.remove(image_path)
        print(f"Deleted image with no matching data: {filename}")
