from ultralytics import YOLO
import os
import requests
import torch

n = 2  # Number of clients
epochs_client = 50
imgsz = 640  # Image size for training
datasets = [
    "/mnt/c/Users/Kevin/FedAvg/dataset/client1",
    "/mnt/c/Users/Kevin/FedAvg/dataset/client2",
]
weights_file_template = 'client_{client_id}_weights.pth'
global_weights_file = 'downloaded_global_weights.pth'

def get_files_in_path(directory_path):
    file_names = []
    for entry in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, entry)):
            file_names.append(os.path.join(directory_path, entry))
    return file_names

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
    model.train(data=dataset_path, epochs=epochs, save=False)
    local_weights = model.state_dict()  # Get the trained model weights
    # Save weights to file
    weights_file = weights_file_template.format(client_id=client_id)
    torch.save(local_weights, + weights_file)
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
        # Train the model locally and save weights to file
        weights_file = train_on_client(models[i], datasets[i], epochs_client, i)
        # Upload the local weights file to the server
        upload_response = upload_weights(i, weights_file)
        if upload_response.get('status') == 'success':
            # Download global weights from the server
            if download_global_weights():
                # Load global weights into the model
                global_weights = torch.load(global_weights_file)
                models[i].load_state_dict(global_weights)
                # Evaluate the updated model
                local_eval = evaluate_model(models[i], datasets[i] + '/test')
                print(f"Client {i} local evaluation after update: {local_eval}")
