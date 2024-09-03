from ultralytics import YOLO
from Fedclient_2 import FedClient
import os
import torch
import matplotlib.pyplot as plt

iterations = 5  # Number of  federation iterations
modelcount = 2 # 2 models
clientId = 1 
epochs_client = 100
imgsz = 640
batch_size = 16
datasets = [
    "/Users/kuangsin/FedAvg/clients/client2/horizon/data.yaml",
    "/Users/kuangsin/FedAvg/clients/client2/top/data.yaml",
]

global_weights_file = 'downloaded_global_weights.pth'
accuracy_trend = []  # Store accuracy for each training

if __name__ == "__main__":
    models = [YOLO('yolov8n.pt') for _ in range(modelcount)]
    fedclient = FedClient()

    for iteration in range(1, iterations + 1):
        iteration_accuracies = []
        for i in range(modelcount):
            weights_file = fedclient.train_on_client(models[i], datasets[i], epochs_client, batch_size, clientId)     
            upload_response = fedclient.upload_weights(clientId, weights_file)
            
            if upload_response.get('status') == 'success':
                if fedclient.download_global_weights():
                    # Load global weights into the model
                    global_weights = torch.load(global_weights_file)
                    models[i].load_state_dict(global_weights)

                    # Evaluate the updated model
                    # local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration, clientId)
                    local_eval = fedclient.evaluate_model(models[i], datasets[i], iteration, clientId)
                    if 'accuracy' in local_eval:
                        iteration_accuracies.append(local_eval['accuracy'])
                    print(f"Client {i} Local evaluation results: {local_eval}")

        # Handling Edge Case: divided by 0 
        if iteration_accuracies:
            accuracy_trend.append(sum(iteration_accuracies) / len(iteration_accuracies))

    # Plot the accuracy trends
    if accuracy_trend:
        plt.plot(range(1, len(accuracy_trend) + 1), accuracy_trend, marker='o')
        plt.title(f"Client {clientId} accuracy trends ( Fo {iterations} Iterations)")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.savefig(f'client_{clientId}_accuracy_trend.png')
        plt.show()
    else:
        print("No accuracy trend data found.")
