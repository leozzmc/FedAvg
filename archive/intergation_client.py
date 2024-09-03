import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from ultralytics import YOLO
import requests

# 提取 ROI 和繪製邊框
def extract_roi(arr, x, y, w, h, intensity, line):
    roi = arr[y:y+h, x:x+w].copy()
    bounding_box = arr.copy()
    cv2.rectangle(bounding_box, (x, y), (x+w, y+h), intensity, line)
    return (roi, bounding_box)

def rotate(origin, point, angle):
    angle = math.radians(angle)
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return round(qx), round(qy)

def process_images(directory):
    output_dir = 'output'
    roi_output_dir = os.path.join(output_dir, 'roi')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(roi_output_dir):
        os.makedirs(roi_output_dir)

    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    for filename in filenames:
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename_without_extension = os.path.splitext(filename)[0]
        df = pd.read_excel('/image_processing/dataset2.xlsx', sheet_name='train')
        pos = df[df['filename'] == filename_without_extension]
        if pos.empty:
            continue
        label = pos['label'].values[0]
        pos = pos[['left_x', 'top_y', 'width', 'height']].values.flatten().tolist()
        ori_x = pos[0]
        ori_y = pos[1]
        leaf_width = pos[2]
        leaf_height = pos[3]
        intensity = (255, 0, 0)  # 用於邊框的紅色
        line = 2  # 邊框的寬度
        roi, bounding_boxed = extract_roi(image, ori_x, ori_y, leaf_width, leaf_height, intensity, line)
        output_path = os.path.join(output_dir, f'output_image_{label}.png')
        plt.imshow(bounding_boxed)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        roi_path = os.path.join(roi_output_dir, 'roi.png')
        plt.imsave(roi_path, roi)

def get_files_in_path(directory_path):
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

def evaluate_model(model, dataset_path):
    results0 = model(get_files_in_path(dataset_path + '/0/'))
    results1 = model(get_files_in_path(dataset_path + '/1/'))
    cnt = 0
    tot = len(results0)
    for result in results0:
        cnt += (result.probs.data[0].item() > 0.5)
    tot += len(results1)
    for result in results1:
        cnt += (result.probs.data[1].item() > 0.5)
    return {"accuracy": cnt / tot}  # Example metric

def train_on_client(model, dataset_path, epochs, client_id):
    process_images(dataset_path)  # Process images before training
    model.train(data=dataset_path, epochs=epochs, save=False)
    local_weights = model.state_dict()  # Get the trained model weights
    weights_file = weights_file_template.format(client_id=client_id)
    torch.save(local_weights, weights_file)
    return weights_file

def upload_weights(client_id, weights_file):
    url = f"http://localhost:5000/api/upload_weights/{client_id}"
    with open(weights_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    if response.status_code != 200:
        print(f"Failed to upload weights for client {client_id}: {response.status_code}")
        print(response.text)
        return {"status": "error"}
    return response.json()

def download_global_weights():
    url = "http://localhost:5000/api/download_global_weights"
    response = requests.get(url)
    if response.status_code == 200:
        with open(global_weights_file, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download global weights: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    models = [YOLO('yolov8n-cls.pt') for _ in range(n)]
    for i in range(n):
        weights_file = train_on_client(models[i], datasets[i], epochs_client, i)
        upload_response = upload_weights(i, weights_file)
        if upload_response.get('status') == 'success':
            if download_global_weights():
                global_weights = torch.load(global_weights_file)
                models[i].load_state_dict(global_weights)
                local_eval = evaluate_model(models[i], datasets[i] + '/test')
                print(f"Client {i} local evaluation after update: {local_eval}")
