import os
import time
import requests
import threading
import csv
import shutil
import cv2  # 导入 OpenCV
from Fedclient import FedClient
from ultralytics import YOLO
from Lib.count_number import count_leaf_number
from Lib.calculate_leaf_area import calculate_leaf_area

iterations = 10  # Number of federation iterations
modelcount = 2  # Number of models (you can modify if more than 1 model is used)
imgsz = 640
large_company_pretrain_params = {
    'epochs': 20,
    'batch': 32,
    'lr0': 0.0001,
    'patience': 3,
    'optimizer': 'SGD',
    'momentum': 0.8,
    'weight_decay': 0.001,
    'dropout': 0.5,
    'augment': True
}
global_weights_file = 'downloaded_global_weights.pth'
accuracy_trend = []  # Store accuracy for each training

def create_model_directory(client_id, model_id):
    """Creates directory for a specific model and client if it doesn't exist."""
    directory = f'model/client_{client_id}/model_{model_id}'
    os.makedirs(directory, exist_ok=True)
    return directory

def create_global_weight_directory():
    """Creates global weight directory for a specific model if it doesn't exist."""
    directory = f'model/global_weight'
    os.makedirs(directory, exist_ok=True)
    return directory

def get_latest_train_directory():
    """Gets the latest train directory from /runs/detect, filtering out directories with large numbers."""
    detect_dir = 'runs/detect'
    train_dirs = [d for d in os.listdir(detect_dir) if d.startswith('train') and d[5:].isdigit() and int(d[5:]) < 50]
    latest_train_dir = max(train_dirs, key=lambda d: os.path.getmtime(os.path.join(detect_dir, d)))
    return os.path.join(detect_dir, latest_train_dir)

def copy_last_weights(client_id):
    """Copies the last.pt file from the latest train directory to the current directory and renames it."""
    latest_train_dir = get_latest_train_directory()
    src = os.path.join(latest_train_dir, 'weights', 'last.pt')
    dst = f'last_{client_id}.pt'
    shutil.copy(src, dst)
    print(f"Copied {src} to {dst}")

def pretrained(client_id, datasets):
    ''' Pre-train the given datasets with YOLOv8 model '''
    print("Pre-training phases...\n")
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()
    for i in range(modelcount):
        if i==0:
            model_dir = create_model_directory(client_id, i)
            #weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i, lr0, patience)
            weights_file = fedclient.train_on_client(models[i], datasets[i], client_id=client_id, model_id=i, training_params=large_company_pretrain_params)
            print(f"Pre-training complete for Model {i}. Weights saved at {weights_file}.")

def retrain_and_evaluate(client_id, datasets, iterations):
    ''' Retrain the model with global weights and calculate accuracy using the test set. '''
    print(f"Client {client_id}: Loading global weights and retraining...\n")
    fedclient = FedClient()
    
    # Copy the last weights to the current directory and rename
    copy_last_weights(client_id)
    
    models = [YOLO(f'last_{client_id}.pt') for _ in range(modelcount)]  # 載入上一次訓練的結果
    accuracies = []

    for model_id in range(modelcount):
        if model_id == 0:
            global_weights = fedclient.download_global_weights(client_id, model_id)
            model_state_dict = models[model_id].state_dict()
        
            try:
                # Remove layers from global weights that don't match model
                filtered_state_dict = {k: v for k, v in global_weights.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
                models[model_id].load_state_dict(filtered_state_dict, strict=False) 
                print(f"Successfully loaded global weights for Model {model_id}.")
            except RuntimeError as e:
                print(f"Error loading global weights for Model {model_id}: {e}")
            
            if input(f"Do you want to retrain Model {model_id}? (y/n): ").lower() == 'y':
                weights_file = fedclient.train_on_client(models[model_id], datasets[model_id], client_id, model_id=model_id, training_params=large_company_pretrain_params)
                print(f"Retraining complete for Model {model_id}. New local weights saved at {weights_file}.")
                
                if input("Do you want to calculate leaf area? (y/n): ").lower() == 'y':
                    # 計算葉片面積
                    leaves_area = {}
                    test_images_dir = "/home/kevin/FedAvg/clients/client1/top/valid/images"
                    for image_file in os.listdir(test_images_dir):
                        if image_file.endswith(".jpg"):
                            image_path = os.path.join(test_images_dir, image_file)
                            areas = calculate_leaf_area(models[model_id], image_path)  # 假設此函數返回葉片面積字典
                            leaves_area.update(areas)

                            # 检查叶片数量
                            if len(areas) > 8:
                                # 读取图像
                                img = cv2.imread(image_path)
                                # 绘制 bounding box
                                for box in models[model_id](img)[0].boxes.data:
                                    x1, y1, x2, y2 = box[:4].int().tolist()
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                # 保存图像
                                error_image_path = f'client_{client_id}_error_handling.jpg'
                                cv2.imwrite(error_image_path, img)
                                print(f"Error image saved at {error_image_path}")

                    # 保存叶片面积到 CSV
                    leaves_area_file = f'{client_id}_leaves_area_{iterations}.csv'
                    with open(leaves_area_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        if os.path.getsize(leaves_area_file) == 0:  # 如果文件为空，写入标题
                            writer.writerow(['ID', 'Area'])
                        for id, area in leaves_area.items():
                            writer.writerow([id, area])  # 写入 ID 和面积

    # 将准确率写入 CSV 文件
    csv_file = f'client_{client_id}_accuracy.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['ModelID', 'Accuracy', 'Iteration'])
        for model_id, accuracy in enumerate(accuracies):
            writer.writerow([model_id, accuracy, iterations])

    print(f"Client {client_id}: Accuracy saved to {csv_file}")


def listen_for_global_weights(client_id, model_id):
    """Listener that waits for the server to notify when global weights are ready."""
    while True:
        print(f"Client {client_id}: Listening for notification for Model {model_id} global weights...")
        response = requests.get(f'http://localhost:5000/api/check_global_weights_status')
        if response.json().get('status') == 'updated':
            print(f"Client {client_id}: Global weights are ready for Model {model_id}. Downloading...")
            fedclient = FedClient()
            create_global_weight_directory()
            if fedclient.download_global_weights(client_id, model_id):
                print(f"Client {client_id}: Global weights downloaded for Model {model_id}.")
            break
        time.sleep(10)  # Wait for 5 seconds before checking again

def main():
    client_id = int(input("Please enter a client ID: "))
    fedclient = FedClient()
    datasets = [
        f"/home/kevin/FedAvg/clients/client{client_id}/top/data.yaml",
        f"/home/kevin/FedAvg/clients/client{client_id}/horizon/data.yaml"
    ]
    
    # Checking if want to pretrain the dataset
    if input("Start pre-trained phases? (y/n): ").lower() == 'y': 
        pretrained(client_id, datasets)
    else: 
        print("Skipping pre-trained phase.\n")

    # Upload weights for both models
    for iter in range(iterations):
        for model_id in range(modelcount):
            if model_id==0:
                weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
                upload_response = fedclient.upload_weights(client_id, model_id, weights_file)

                if upload_response.get('status') == 'pending':
                    listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
                    listener_thread.start()
            
        # Retrain and evaluate after receiving global weights
        retrain_and_evaluate(client_id, datasets, iter)

if __name__ == "__main__":
    main()
