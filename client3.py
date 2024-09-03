from ultralytics import YOLO
from Fedclient import FedClient
import os, time, requests
import torch
import matplotlib.pyplot as plt


iterations = 5  # Number of federation iterations
modelcount = 2  # 2 models
epochs_client = 100
imgsz = 640
batch_size = 16


global_weights_file = 'downloaded_global_weights.pth'
accuracy_trend = []  # Store accuracy for each training

if __name__ == "__main__":
    
    fedclient = FedClient()
    
    
    # Pre-trained phases, train for two models
    clientId = int(input("Input client ID:  "))
    
    datasets = [
        f"/Users/kuangsin/FedAvg/clients/client{clientId}/horizon/data.yaml",
        f"/Users/kuangsin/FedAvg/clients/client{clientId}/top/data.yaml",
    ]
    
    if input("Start pre-trained phases? (y/n): ").lower() == 'y':
        print("Pre-training phases...\n")
        models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
        for i in range(modelcount):
            fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)
    else:
        print("Skip pre-trained phase.\n")
    
    # Federation Learning phases
    print("Federated Learning phases...\n")
    models = [YOLO('last.pt'), YOLO('above_last.pt')]
    
    for iteration in range(1, iterations + 1):
        iteration_accuracies = []
        if input(f'Upload weights for client {clientId}? (y/n): ').lower() == 'y':
            weights_file = f'client_{clientId}_weights.pth'
            upload_response = fedclient.upload_weights(clientId, weights_file)
        else:
            break

        # Automatically handle pending status
        while upload_response.get('status') == 'pending':
            print("Waiting for other clients to upload their weights...")
            time.sleep(10)  # Wait for 10 seconds before checking again
            download_response = fedclient.download_global_weights()
            if download_response:
                print("Downloading global weights\n")
                for i in range(modelcount):
                        # Load global weights into the model
                        global_weights = torch.load(global_weights_file)
                        models[i].load_state_dict(global_weights)

                        # Retrain with the updated global weights
                        print(f"Retraining Model {i}...\n")
                        fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)

                        # Evaluate the updated model
                        local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration, clientId)
                        if 'accuracy' in local_eval:
                            iteration_accuracies.append(local_eval['accuracy'])
                            print(f"Client {clientId} - Model {i} Local evaluation results: {local_eval}")

                        # Save training results
                        csv_filename = f"client_{clientId}_training_results.csv"
                        with open(csv_filename, 'w') as f:
                            f.write("iterations,accuracy\n")
                            for i, accuracy in enumerate(iteration_accuracies, start=1):
                                f.write(f"{i},{accuracy}\n")
                        print(f"Saved training results to {csv_filename}")
            break

        
        if upload_response.get('status') == 'success':
            print("Successfully uploaded\n")
            print("Waiting for global weights...\n")
            if fedclient.check_server_for_global_weights():
                print("Downloading global weights\n")
                if fedclient.download_global_weights():
                    for i in range(modelcount):
                        # Load global weights into the model
                        
                        global_weights = torch.load(global_weights_file)
                        models[i].load_state_dict(global_weights)

                        # Retrain with the updated global weights
                        print(f"Retraining Model {i}...\n")
                        fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)

                        # Evaluate the updated model
                        local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration, clientId)
                        if 'accuracy' in local_eval:
                            iteration_accuracies.append(local_eval['accuracy'])
                            print(f"Client {clientId} - Model {i} Local evaluation results: {local_eval}")

                    # Save training results
                    csv_filename = f"client_{clientId}_training_results.csv"
                    with open(csv_filename, 'w') as f:
                        f.write("iterations,accuracy\n")
                        for i, accuracy in enumerate(iteration_accuracies, start=1):
                            f.write(f"{i},{accuracy}\n")
                    print(f"Saved training results to {csv_filename}")
