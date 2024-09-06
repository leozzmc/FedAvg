import os
import time
import requests
from new_Fedclient import FedClient
from ultralytics import YOLO

iterations = 5  # Number of federation iterations
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

def pretrained(client_id, datasets):
    ''' Pre-train the given datasets with YOLOv8 model '''
    print("Pre-training phases...\n")
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()
    for i in range(modelcount):
        model_dir = create_model_directory(client_id, i)
        weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, client_id, i)
        print(f"Pre-training complete for Model {i}. Weights saved at {weights_file}.")

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

        while upload_response.get('status') == 'pending':
            print(f"Waiting for other clients to upload weights for Model {model_id}...")
            time.sleep(10)
            status_response = fedclient.check_global_weights_status()
            if status_response.get('status') == 'updated':
                print(f"Global weights updated for Model {model_id}. Downloading...\n")
                if fedclient.download_global_weights(client_id, model_id):
                    print(f"Global weights downloaded for Model {model_id}.")
                break
            else:
                print(f"Global weights not updated yet for Model {model_id}. Continuing to wait...")

if __name__ == "__main__":
    main()
