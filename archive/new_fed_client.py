from ultralytics import YOLO
import os
import torch
import requests

# Configuration parameters
epochs_client = 50  # Number of epochs for each client's training
datasets = [
    "/mnt/c/Users/Kevin/FedAvg/dataset/client1",
    "/mnt/c/Users/Kevin/FedAvg/dataset/client2",
]
model_path = 'yolov8n-cls.pt'  # Path to the model

# Initialize the model
model = YOLO(model_path)

def get_files_in_path(directory_path):
    """Retrieve all file paths in a given directory."""
    file_names = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            file_names.append(full_path)
    return file_names

def train_on_client(model, dataset_path, epochs):
    """Train the model on a client dataset."""
    data_config = os.path.join(dataset_path, 'data.yaml')  # Ensure the data.yaml file is in the dataset directory
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Configuration file {data_config} not found.")
    
    model.train(data=data_config, epochs=epochs, save=False)
    return model.state_dict()  # Return the model weight after training

def upload_weights(weights, server_url):
    """Upload model weights to the server."""
    weights_path = 'local_weights.pth'
    torch.save(weights, weights_path)  # Save weights to a file
    
    with open(weights_path, 'rb') as f:
        response = requests.post(server_url, files={'file': f})
    
    return response.text

# Train the model and upload weights for each client dataset
for i, path in enumerate(datasets):
    print(f"Training on dataset {path}...")
    try:
        local_weights = train_on_client(model, path, epochs_client)
        print(f"Training complete for dataset {path}.")
        
        print(f"Uploading weights for dataset {path}...")
        server_response = upload_weights(local_weights, 'http://127.0.0.1:5000/upload_weights')
        print(f"Server response for dataset {path}: {server_response}")
    except Exception as e:
        print(f"An error occurred for dataset {path}: {e}")
