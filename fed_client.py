from ultralytics import YOLO
import os
import torch
import requests 

# Config parameters
epochs_client = 50  # 每個客戶端的訓練輪次
datasets = [
    "/mnt/c/Users/Kevin/FedAvg/dataset/client1",
    "/mnt/c/Users/Kevin/FedAvg/dataset/client2",
]
model_path = 'yolov8n-cls.pt'  # Model path


# Init the model
model = YOLO(model_path)

def get_files_in_path(directory_path):
    """Retrieve all file paths in a given directory."""
    file_names = []
    for entry in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, entry)):
            file_names.append(os.path.join(directory_path, entry))
    return file_names

def train_on_client(model, dataset_path, epochs):
    """Train the model on a client dataset.。"""
    model.train(data=dataset_path, epochs=epochs, save=False)
    return model.state_dict()  # Return the model weight after training



## Train the model

for path in datasets:
    local_weights = train_on_client(model, path, epochs_client)

## Upload the weight to the server

weights_path = 'local_weights.pth'
torch.save(local_weights, weights_path)  ## ???


## Send requests to the server
with open(weights_path, 'rb') as f:
    response = requests.post('http://127.0.0.1:5000/upload_weights', files={'file': f})

print(f"Server response: {response.text}")