from ultralytics import YOLO
import Fedclient
import Fedclient_new
import os
import requests
import torch

iterations = 5
modelcount = 2  
clientId = 0
epochs_client = 100
imgsz = 640 
batch_size = 16 
datasets = [
    "/Users/kuangsin/FedAvg/clients/client1/horizon/data.yaml",
    "/Users/kuangsin/FedAvg/clients/client1/top/data.yaml",
]

#eights_file_template = 'client_{client_id}_weights.pth'
global_weights_file = 'downloaded_global_weights.pth'

if __name__ == "__main__":
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = Fedclient_new.FedClient()

    for iteration in range(1, iterations + 1):
        for i in range(modelcount):
            # Train the model locally and save weights to file
            weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)
            # Upload the local weights file to the server
            upload_response = fedclient.upload_weights(i, weights_file)
            if upload_response.get('status') == 'success':
                # Download global weights from the server
                if fedclient.download_global_weights():
                    # Load global weights into the model
                    global_weights = torch.load(global_weights_file)
                    models[i].load_state_dict(global_weights)
                    # Evaluate the updated model and update accuracy trend plot
                    local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration)
                    print(f"Client {i} local evaluation after update: {local_eval}")


