# import os
# import time
# import requests
# import threading
# from new_Fedclient import FedClient
# from ultralytics import YOLO

# iterations = 1  # Number of federation iterations
# modelcount = 1  # 2 models
# epochs_client = 100
# imgsz = 640
# batch_size = 16
# global_weights_file = 'downloaded_global_weights.pth'
# accuracy_trend = []  # Store accuracy for each training

# def create_model_directory(client_id, model_id):
#     """Creates directory for a specific model and client if it doesn't exist."""
#     directory = f'model/client_{client_id}/model_{model_id}'
#     os.makedirs(directory, exist_ok=True)
#     return directory

# def create_global_weight_directory():
#     """Creates global weight directory for a specific model if it doesn't exist."""
#     directory = f'model/global_weight'
#     os.makedirs(directory, exist_ok=True)
#     return directory

# def pretrained(client_id, datasets):
#     ''' Pre-train the given datasets with YOLOv8 model '''
#     print("Pre-training phases...\n")
#     models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
#     fedclient = FedClient()
#     for i in range(modelcount):
#         model_dir = create_model_directory(client_id, i)
#         weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i)
#         print(f"Pre-training complete for Model {i}. Weights saved at {weights_file}.")

# def listen_for_global_weights(client_id, model_id):
#     """Listener that waits for the server to notify when global weights are ready."""
#     while True:
#         print(f"Client {client_id}: Listening for notification for Model {model_id} global weights...")
#         response = requests.get(f'http://localhost:5000/api/check_global_weights_status')
#         if response.json().get('status') == 'updated':
#             print(f"Client {client_id}: Global weights are ready for Model {model_id}. Downloading...")
#             fedclient = FedClient()
#             create_global_weight_directory()
#             if fedclient.download_global_weights(client_id, model_id):
#                 print(f"Client {client_id}: Global weights downloaded for Model {model_id}.")
#             break
#         time.sleep(5)  # Wait for 5 seconds before checking again

# def main():
#     client_id = int(input("Please enter a client ID: "))
#     fedclient = FedClient()
    
#     datasets = [
#         f"/mnt/c/Users/Kevin/FedAvg/clients/client{client_id}/horizon/data.yaml",
#     ]

#     #    f"/mnt/c/Users/Kevin/FedAvg/clients/client{client_id}/top/data.yaml",
    
    
#     # Checking if want to pretrain the dataset
#     if input("Start pre-trained phases? (y/n): ").lower() == 'y': 
#         pretrained(client_id, datasets)
#     else: 
#         print("Skipping pre-trained phase.\n")

#     # Upload weights for both models
#     for model_id in range(modelcount):
#         weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
#         upload_response = fedclient.upload_weights(client_id, model_id, weights_file)

#         if upload_response.get('status') == 'pending':
#             # 使用新線程來監聽全局權重通知
#             listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
#             listener_thread.start()
        

# if __name__ == "__main__":
#     main()

#### INTEGRATED WITH OUTPUT ACCURACY FUNCTIONALITY ######

# import os
# import time
# import requests
# import threading
# import csv
# from new_Fedclient import FedClient
# from ultralytics import YOLO

# iterations = 1  # Number of federation iterations
# modelcount = 1  # 2 models
# epochs_client = 100
# imgsz = 640
# batch_size = 16
# global_weights_file = 'downloaded_global_weights.pth'
# accuracy_trend = []  # Store accuracy for each training

# def create_model_directory(client_id, model_id):
#     """Creates directory for a specific model and client if it doesn't exist."""
#     directory = f'model/client_{client_id}/model_{model_id}'
#     os.makedirs(directory, exist_ok=True)
#     return directory

# def create_global_weight_directory():
#     """Creates global weight directory for a specific model if it doesn't exist."""
#     directory = f'model/global_weight'
#     os.makedirs(directory, exist_ok=True)
#     return directory

# def pretrained(client_id, datasets):
#     ''' Pre-train the given datasets with YOLOv8 model '''
#     print("Pre-training phases...\n")
#     models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
#     fedclient = FedClient()
#     for i in range(modelcount):
#         model_dir = create_model_directory(client_id, i)
#         weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i)
#         print(f"Pre-training complete for Model {i}. Weights saved at {weights_file}.")

# def retrain_and_evaluate(client_id, datasets):
#     ''' Retrain the model with global weights and calculate accuracy '''
#     print(f"Client {client_id}: Loading global weights and retraining...\n")
#     fedclient = FedClient()
    
#     models = [YOLO('yolov8n.pt') for _ in range(modelcount)]  # 初始化模型
#     accuracies = []

#     for model_id in range(modelcount):
#         global_weights = fedclient.download_global_weights(client_id, model_id)
    
#         try:
#             models[model_id].load_state_dict(global_weights, strict=False)  # 忽略不匹配部分
#             print(f"Successfully loaded global weights for Model {model_id}.")
#         except RuntimeError as e:
#             print(f"Error loading global weights for Model {model_id}: {e}")
        
#         # 提示是否進行 retrain
#         if input(f"Do you want to retrain Model {model_id}? (y/n): ").lower() == 'y':
#             weights_file = fedclient.train_on_client(models[model_id], datasets[model_id], epochs_client, batch_size, client_id, model_id)
#             print(f"Retraining complete for Model {model_id}. New local weights saved at {weights_file}.")
            
#             # 計算 accuracy 並保存
#             accuracy_data = fedclient.evaluate_model(models[model_id], datasets[model_id], iterations, client_id)
#             accuracies.append(accuracy_data['accuracy'])
            
#     # 保存 accuracy 到 CSV 檔案
#     csv_file = f'client_{client_id}_accuracy.csv'
#     with open(csv_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['ModelID', 'Accuracy', 'Iteration'])
#         for model_id, accuracy in enumerate(accuracies):
#             writer.writerow([model_id, accuracy, iterations])

#     print(f"Client {client_id}: Accuracy saved to {csv_file}")

# def listen_for_global_weights(client_id, model_id):
#     """Listener that waits for the server to notify when global weights are ready."""
#     while True:
#         print(f"Client {client_id}: Listening for notification for Model {model_id} global weights...")
#         response = requests.get(f'http://localhost:5000/api/check_global_weights_status')
#         if response.json().get('status') == 'updated':
#             print(f"Client {client_id}: Global weights are ready for Model {model_id}. Downloading...")
#             fedclient = FedClient()
#             create_global_weight_directory()
#             if fedclient.download_global_weights(client_id, model_id):
#                 print(f"Client {client_id}: Global weights downloaded for Model {model_id}.")
#             break
#         time.sleep(5)  # Wait for 5 seconds before checking again

# def main():
#     client_id = int(input("Please enter a client ID: "))
#     fedclient = FedClient()
    
#     datasets = [
#         f"/mnt/c/Users/Kevin/FedAvg/clients/client{client_id}/horizon/data.yaml",
#     ]

#     #    f"/mnt/c/Users/Kevin/FedAvg/clients/client{client_id}/top/data.yaml",
    
#     # Checking if want to pretrain the dataset
#     if input("Start pre-trained phases? (y/n): ").lower() == 'y': 
#         pretrained(client_id, datasets)
#     else: 
#         print("Skipping pre-trained phase.\n")

#     # Upload weights for both models
#     for model_id in range(modelcount):
#         weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
#         upload_response = fedclient.upload_weights(client_id, model_id, weights_file)

#         if upload_response.get('status') == 'pending':
#             # 使用新線程來監聽全局權重通知
#             listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
#             listener_thread.start()
    
#     # Retrain and evaluate after receiving global weights
#     retrain_and_evaluate(client_id, datasets)


# if __name__ == "__main__":
#     main()

## Modification 2 ####

import os
import time
import requests
import threading
import csv
from new_Fedclient import FedClient
from ultralytics import YOLO

iterations = 5  # Number of federation iterations
modelcount = 1  # Number of models
epochs_client = 100
imgsz = 640
batch_size = 16
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

def save_accuracy_to_csv(client_id, accuracies, iteration):
    ''' Save the accuracy results to CSV file '''
    csv_file = f"client_{client_id}_accuracy.csv"
    
    # 检查文件是否存在，如果不存在则创建文件并写入表头
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # 如果文件不存在，写入表头
            writer.writerow(["Iteration", "ModelID", "Accuracy"])
        
        for model_id, accuracy in enumerate(accuracies):
            # 检查 accuracy 数据是否为空或 None
            if accuracy is not None:
                writer.writerow([iteration, model_id, accuracy])
            else:
                print(f"Accuracy for Model {model_id} in iteration {iteration} is empty. Skipping.")
    
    print(f"Accuracy for iteration {iteration} has been written to {csv_file}.")

def pretrained(client_id, datasets):
    ''' Pre-train the given datasets with YOLOv8 model '''
    print("Pre-training phases...\n")
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()
    for i in range(modelcount):
        model_dir = create_model_directory(client_id, i)
        weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i)
        print(f"Pre-training complete for Model {i}. Weights saved at {weights_file}.")

def download_global_weights_with_wait(client_id, model_id, max_retries=10, wait_time=30):
    ''' Attempt to download global weights, retrying until available '''
    fedclient = FedClient()
    for attempt in range(max_retries):
        success = fedclient.download_global_weights(client_id, model_id)
        if success:
            print(f"Successfully downloaded global weights for Model {model_id} in iteration {attempt + 1}.")
            return success
        else:
            print(f"Global weights not available for Model {model_id}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
    print(f"Max retries reached. Skipping retraining for Model {model_id}.")
    return None


def retrain_and_evaluate(client_id, datasets, iteration):
    """ Retrain the model with global weights and calculate accuracy """
    print(f"Client {client_id}: Loading global weights and retraining for iteration {iteration}...\n")
    fedclient = FedClient()
    
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]  # Initialize models
    accuracies = []

    for model_id in range(modelcount):
        global_weights = fedclient.download_global_weights(client_id, model_id)
    
        try:
            models[model_id].load_state_dict(global_weights, strict=False)  # Ignore mismatched parts
            print(f"Successfully loaded global weights for Model {model_id}.")
        except RuntimeError as e:
            print(f"Error loading global weights for Model {model_id}: {e}")
            continue  # Skip this model if weights loading failed
        
        # Prompt if retrain is needed
        if input(f"Do you want to retrain Model {model_id}? (y/n): ").lower() == 'y':
            weights_file = fedclient.train_on_client(models[model_id], datasets[model_id], epochs_client, batch_size, client_id, model_id)
            print(f"Retraining complete for Model {model_id}. New local weights saved at {weights_file}.")
            
            # Calculate accuracy and save
            accuracy_data = fedclient.evaluate_model(models[model_id], datasets[model_id], iterations, client_id)
            accuracy = accuracy_data['accuracy']
            print(f"Accuracy for Model {model_id} in iteration {iteration}: {accuracy}")
            accuracies.append((model_id, accuracy, iteration))
        else:
            print(f"Skipping retrain for Model {model_id}.")

    # Save accuracy to CSV file
    csv_file = f'client_{client_id}_accuracy.csv'
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ModelID', 'Accuracy', 'Iteration'])
        for model_id, accuracy, iteration in accuracies:
            writer.writerow([model_id, accuracy, iteration])

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
        time.sleep(15)  # Wait for 5 seconds before checking again

def main():
    client_id = int(input("Please enter a client ID: "))
    fedclient = FedClient()
    
    datasets = [
        f"/mnt/c/Users/Kevin/FedAvg/clients/client{client_id}/horizon/data.yaml",
    ]

    # CSV 檔案
    csv_file = f'client_{client_id}_accuracy.csv'

    # Checking if want to pretrain the dataset
    if input("Start pre-trained phases? (y/n): ").lower() == 'y': 
        pretrained(client_id, datasets)
    else: 
        print("Skipping pre-trained phase.\n")

    # Loop for multiple iterations of federated learning
    for iteration in range(1, iterations + 1):
        print(f"===== Starting iteration {iteration} =====")

        # Upload weights for both models
        for model_id in range(modelcount):
            weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
            upload_response = fedclient.upload_weights(client_id, model_id, weights_file)

            if upload_response.get('status') == 'pending':
                # 使用新線程來監聽全局權重通知
                listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
                listener_thread.start()

        # Retrain and evaluate after receiving global weights
        accuracies = retrain_and_evaluate(client_id, datasets, iteration)

        # Append accuracy to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for model_id, accuracy in enumerate(accuracies):
                writer.writerow([model_id, accuracy, iteration])

    print(f"Client {client_id}: Federated learning completed. Accuracy saved to {csv_file}")

if __name__ == "__main__":
    main()

