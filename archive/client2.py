import os
import time
import requests
import threading
import csv
import shutil
from Fedclient import FedClient
from ultralytics import YOLO

iterations = 10  # Number of federation iterations
modelcount = 2  # Number of models (you can modify if more than 1 model is used)
epochs_client = 5
imgsz = 640
batch_size = 8
lr0 = 0.01
patience = 3
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
            weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i, lr0, patience)
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
                weights_file = fedclient.train_on_client(models[model_id], datasets[model_id], epochs_client, batch_size, client_id, model_id, lr0, patience)
                print(f"Retraining complete for Model {model_id}. New local weights saved at {weights_file}.")
                
                print(f"Evaluating accuracy on the test set for Model {model_id}...")
                accuracy_data = fedclient.evaluate_model(models[model_id], datasets[model_id], iterations, client_id)
                accuracies.append(accuracy_data['accuracy'])

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
        f"/Users/kuangsin/FedAvg/clients/client{client_id}/horizon/data.yaml",
        f"/Users/kuangsin/FedAvg/clients/client{client_id}/top/data.yaml"
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
