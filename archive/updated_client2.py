
from ultralytics import YOLO
from archive.Fedclient_new import FedClient
import os
import torch
import matplotlib.pyplot as plt

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
accuracy_trend = []  # Store accuracy for each iteration

if __name__ == "__main__":
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()

    for iteration in range(1, iterations + 1):
        iteration_accuracies = []
        for i in range(modelcount):
            # Train the model locally and save weights to file
            weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)
            # weights_file =  "/client_weights/client_1_weights.pth"
            
            # Upload the local weights file to the server
            upload_response = fedclient.upload_weights(clientId, weights_file)
            
            if upload_response.get('status') == 'success':
                # Download global weights from the server
                if fedclient.download_global_weights():
                    # Load global weights into the model
                    global_weights = torch.load(global_weights_file)
                    models[i].load_state_dict(global_weights)

                    # Evaluate the updated model
                    local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration, clientId)
                    if 'accuracy' in local_eval:
                        iteration_accuracies.append(local_eval['accuracy'])
                    print(f"Client {i} local evaluation after update: {local_eval}")

        # Avoid division by zero by checking if iteration_accuracies is not empty
        if iteration_accuracies:
            accuracy_trend.append(sum(iteration_accuracies) / len(iteration_accuracies))
        else:
            print(f"No accuracy recorded for iteration {iteration}, skipping.")

    # Plot the accuracy trend after iterations
    if accuracy_trend:
        plt.plot(range(1, len(accuracy_trend) + 1), accuracy_trend, marker='o')
        plt.title(f"Client {clientId} Accuracy Trend Across {iterations} Iterations")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.savefig(f'client_{clientId}_accuracy_trend.png')
        plt.show()
    else:
        print("No accuracy data available to plot.")
