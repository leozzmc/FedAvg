import os
import time
import requests
import threading
import csv
import shutil
from Fedclient import FedClient
from ultralytics import YOLO
from Lib.count_number import count_leaf_number

iterations = 10  # Number of federation iterations
modelcount = 2  # Number of models (you can modify if more than 1 model is used)
imgsz = 640
# large_company_pretrain_params = {
#     'epochs': 20,
#     'batch': 64,
#     'lr0': 0.0005,
#     'patience': 5,
#     'momentum': 0.9,
#     'optimizer': 'SGD',
#     'weight_decay': 0.001,
#     'dropout': 0.2,
#     'augment': True
# }
large_company_pretrain_params = {
    'epochs': 12,  # 減少 epochs
    'batch': 32,   # 減少 batch size
    'lr0': 0.0005, # 保持不變
    'patience': 5, # 保持不變
    'momentum': 0.9, # 保持不變
    'optimizer': 'SGD', # 保持不變
    'weight_decay': 0.001, # 保持不變
    'dropout': 0.3, # 增加 dropout
    'augment': True # 保持不變
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
    models = [YOLO('yolov8s.pt') for _ in range(modelcount)]
    fedclient = FedClient()
    for i in range(modelcount):
        if i==1:
            model_dir = create_model_directory(client_id, i)
            #weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i, lr0, patience)
            weights_file = fedclient.train_on_client(models[i], datasets[i], client_id=1, model_id=1, training_params=large_company_pretrain_params)
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
        if model_id == 1:
            global_weights = fedclient.download_global_weights(client_id, model_id)
            model_state_dict = models[model_id].state_dict()
        
            try:
                # Remove layers from global weights that don't match model
                filtered_state_dict = {k: v for k, v in global_weights.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
                models[model_id].load_state_dict(filtered_state_dict, strict=False) 
                print(f"Model's first layer weights after loading global weights: {list(models[model_id].parameters())[0]}")
                print(f"Successfully loaded global weights for Model {model_id}.")
            except RuntimeError as e:
                print(f"Error loading global weights for Model {model_id}: {e}")
            
            if input(f"Do you want to retrain Model {model_id}? (y/n): ").lower() == 'y':
                weights_file = fedclient.train_on_client(models[model_id], datasets[model_id], client_id, model_id=1, training_params=large_company_pretrain_params)
                print(f"Retraining complete for Model {model_id}. New local weights saved at {weights_file}.")
                
                print(f"Evaluating accuracy on the test set for Model {model_id}...")
                accuracy_data = fedclient.evaluate_model(models[model_id], datasets[model_id], iterations, client_id)
                accuracies.append(accuracy_data['accuracy'])

                # 新增預測步驟
                if model_id == 1:
                    if input("Do you want to make predictions? (y/n): ").lower() == 'y':
                        # 進行預測
                        test_images_dir = f"horizon/test/images/"
                        actual_numbers = {}

                        # 讀取實際葉片數量
                        with open('archive/orchid_actual_number.csv', mode='r') as file:
                            reader = csv.DictReader(file)
                            for row in reader:
                                actual_numbers[row['ID']] = int(row['Leave Numbers'])

                        print("Actual numbers:", actual_numbers)  # 檢查實際數量

                        predictions = {}
                        for image_id in actual_numbers.keys():
                            image_path = os.path.join(test_images_dir, f"{image_id}*.jpg")  # 假設圖片格式為 .jpg
                            predicted_count = count_leaf_number(models[model_id], image_path)  # 將模型傳遞給函數
                            predictions[image_id] = predicted_count

                        print("Predictions:", predictions)  # 檢查預測結果

                        # 計算絕對誤差
                        absolute_errors = {id: abs(predictions[id] - actual_numbers[id]) for id in predictions}
                        print("Absolute errors:", absolute_errors)  # 檢查絕對誤差

                        # 計算平均絕對誤差 (MAE)
                        mae = sum(absolute_errors.values()) / len(absolute_errors) if absolute_errors else 0
                        print(f"Mean Absolute Error (MAE): {mae}")

                        # 保存結果到 horizon_mae.csv
                        with open(f'horizon_mae_{client_id}.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            if file.tell() == 0:  # 檢查是否為空文件
                                writer.writerow(['Iteration', 'Mean Absolute Error (MAE)'])
                            writer.writerow([iterations, mae])  # 寫入當前迭代和 MAE


    # 將準確率寫入 CSV 文件
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
            if model_id==1:
                weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
                upload_response = fedclient.upload_weights(client_id, model_id, weights_file)

                if upload_response.get('status') == 'pending':
                    listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
                    listener_thread.start()
            
        # Retrain and evaluate after receiving global weights
        retrain_and_evaluate(client_id, datasets, iter)

if __name__ == "__main__":
    main()
