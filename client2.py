from ultralytics import YOLO
# import Fedclient
from Fedclient_new import FedClient
import os
import requests
import torch

iterations = 5
modelcount = 2  
clientId = 1
epochs_client = 100
imgsz = 640 
batch_size = 16 
datasets = [
    "/mnt/c/Users/Kevin/FedAvg/clients/client2/horizon/data.yaml",
    "/mnt/c/Users/Kevin/FedAvg/clients/client2/top/data.yaml",
]


global_weights_file = 'downloaded_global_weights.pth'

if __name__ == "__main__":
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()

    for iteration in range(1, iterations + 1):
        for i in range(modelcount):
            # Train the model locally and save weights to file
            # weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)
            weights_file = "client_1_weights.pth"
            # Upload the local weights file to the server

            upload_response = fedclient.upload_weights(clientId, weights_file)
            if upload_response.get('status') == 'success':
                input("Are yiu sure to download global weights? (y/n): ")  
                # Download global weights from the server
                if fedclient.download_global_weights():
                    # Load global weights into the model
                    input("Are you sure to evaluate the model? (y/n): ")
                    global_weights = torch.load(global_weights_file)
                    models[i].load_state_dict(global_weights)
                    # Evaluate the updated model and update accuracy trend plot
                    local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration)
                    print(f"Client {i} local evaluation after update: {local_eval}")
            
            # upload_response = fedclient.upload_weights(clientId, weights_file)

            # # Polling or waiting mechanism to handle 'pending' status
            # max_retries = 10
            # retry_count = 0

            # while upload_response.get('status') == 'pending' and retry_count < max_retries:
            #     print(f"Upload pending, retrying... ({retry_count + 1}/{max_retries})")
            #     time.sleep(5)  # Wait for 5 seconds before retrying (adjust as needed)
            #     upload_response = fedclient.upload_weights(clientId, weights_file)
            #     retry_count += 1

            # if upload_response.get('status') == 'success':
            #     # Download global weights from the server
            #     if fedclient.download_global_weights():
            #         # Load global weights into the model
            #         global_weights = torch.load(global_weights_file)
            #         models[i].load_state_dict(global_weights)
            # else:
            #     print("Upload failed or did not succeed after retries.")

                
