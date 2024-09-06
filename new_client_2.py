import os
import time
import requests
import threading
from new_Fedclient import FedClient
from ultralytics import YOLO

iterations = 1  # Number of federation iterations
modelcount = 2  # 2 models
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

def pretrained(client_id, datasets):
    ''' Pre-train the given datasets with YOLOv8 model '''
    print("Pre-training phases...\n")
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()
    for i in range(modelcount):
        model_dir = create_model_directory(client_id, i)
        weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i)
        print(f"Pre-training complete for Model {i}. Weights saved at {weights_file}.")

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
        time.sleep(5)  # Wait for 5 seconds before checking again

def main():
    client_id = int(input("Please enter a client ID: "))
    fedclient = FedClient()
    
    datasets = [
        f"/Users/kuangsin/FedAvg/clients/client{client_id}/horizon/data.yaml",
        f"/Users/kuangsin/FedAvg/clients/client{client_id}/top/data.yaml",
    ]
    
    # Checking if want to pretrain the dataset
    if input("Start pre-trained phases? (y/n): ").lower() == 'y': 
        pretrained(client_id, datasets)
    else: 
        print("Skipping pre-trained phase.\n")

    # Upload weights for both models
    for model_id in range(modelcount):
        weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
        upload_response = fedclient.upload_weights(client_id, model_id, weights_file)

        if upload_response.get('status') == 'pending':
            # 使用新線程來監聽全局權重通知
            listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
            listener_thread.start()
        

if __name__ == "__main__":
    main()

#### INTEGRATED WITH OUTPUT ACCURACY FUNCTIONALITY ######
# import os
# import time
# import requests
# import threading
# import torch
# from ultralytics import YOLO
# from new_Fedclient import FedClient

# iterations = 1
# modelcount = 2
# epochs_client = 100
# batch_size = 16
# global_weights_file = 'model/global_weight/global_weights.pth'
# accuracy_trend = [] 



# def create_model_directory(client_id, model_id):
#     """Creates directory for a specific model and client if it doesn't exist."""
#     directory = f'model/client_{client_id}/model_{model_id}'
#     os.makedirs(directory, exist_ok=True)
#     return directory

# def create_global_weight_directory():
#     """Creates global weight directory if it doesn't exist."""
#     directory = 'model/global_weight'
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

# def save_global_weights(model_id, global_weights):
#     """Save global weights to the specified directory."""
#     global_weights_dir = f'model/global_weight/model_{model_id}/'
#     os.makedirs(global_weights_dir, exist_ok=True)  
#     global_weights_file = os.path.join(global_weights_dir, f'global_weights_client_model_{model_id}.pth')
    
#     torch.save(global_weights, global_weights_file)  
#     print(f"Global weights for Model {model_id} saved at {global_weights_file}")
    
# def listen_for_global_weights(client_id, model_id):
#     """Listener that waits for the server to notify when global weights are ready."""
#     fedclient = FedClient()  # 初始化 FedClient 以供下載權重使用
#     while True:
#         print(f"Client {client_id}: Listening for notification for Model {model_id} global weights...")
#         response = requests.get(f'http://localhost:5000/api/check_global_weights_status')
#         if response.json().get('status') == 'updated':
#             print(f"Client {client_id}: Global weights are ready for Model {model_id}. Downloading...")
#             global_weights = fedclient.download_global_weights(client_id, model_id)  # 下載全局權重

#             if global_weights:
#                 print(f"Client {client_id}: Global weights downloaded for Model {model_id}.")
#                 save_global_weights(model_id, global_weights)  # 保存全局權重
#                 break  # 全局權重已下載並保存，退出循環
#         time.sleep(10)  # 等待 10 秒後再檢查狀態

# def main():
#     client_id = int(input("Please enter a client ID: "))
#     fedclient = FedClient()
    
#     datasets = [
#         f"/Users/kuangsin/FedAvg/clients/client{client_id}/horizon/data.yaml",
#         f"/Users/kuangsin/FedAvg/clients/client{client_id}/top/data.yaml",
#     ]
    
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
#             # Use a new thread to listen for global weights notification
#             listener_thread = threading.Thread(target=listen_for_global_weights, args=(client_id, model_id))
#             listener_thread.start()
        
#             # Wait for the global weights to be available and download them
#             listener_thread.join()

#         global_weights = fedclient.download_global_weights(client_id, model_id)
#         if global_weights is None:
#             print(f"Error: Global weights for Model {model_id} could not be downloaded")
#             continue

#         # Initialize the model before loading the global weights
#         model = YOLO('yolov8n.pt')  # Initialize the model for each model_id
        
#         try:
#             model.load_state_dict(global_weights)  # Load the correct global weights
#         except TypeError as e:
#             print(f"Error: Could not load global weights for Model {model_id} - {e}")
#             continue

#         # Retrain with the updated global weights
#         print(f"Retraining Model {model_id}...\n")
#         fedclient.train_on_client(model, datasets[model_id], epochs_client, batch_size, client_id)

#         # Evaluate the updated model
#         local_eval = fedclient.evaluate_model(model, datasets[model_id], client_id)
#         if 'accuracy' in local_eval:
#             accuracy_trend.append({
#                 'modelID': model_id,
#                 'accuracy': local_eval['accuracy'],
#                 'iteration': iterations
#             })
#             print(f"Client {client_id} - Model {model_id} Local evaluation results: {local_eval}")
    
#     # Save training results to a CSV file
#     csv_filename = f"client_{client_id}_training_results.csv"
#     with open(csv_filename, 'w') as f:
#         f.write("modelID,accuracy,iteration\n")
#         for entry in accuracy_trend:
#             f.write(f"{entry['modelID']},{entry['accuracy']},{entry['iteration']}\n")
#     print(f"Saved training results to {csv_filename}")

    


# if __name__ == "__main__":
#     main()
