import requests
import argparse
import os
from ultralytics import YOLO
from Fedclient import FedClient
import torch

epochs= 100
batch_size = 16
def train_model(models, datasets, epochs_client, batch_size, client_id):
    """
    Function to train the models on the client.
    """
    local_weights_files = []
    for i, model in enumerate(models):
        local_weights_file = fedclient.train_on_client(
            model, datasets[i], epochs_client, batch_size, client_id
        )
        local_weights_files.append(local_weights_file)
    return local_weights_files

def upload_local_weights(local_weights_files, client_id):
    """
    Function to upload local weights to the server.
    """
    for weights_file in local_weights_files:
        upload_response = fedclient.upload_weights(client_id, weights_file)
        if upload_response.get('status') != 'success':
            print(f"Failed to upload weights for client {client_id}")
            return False
    return True

def download_global_weights(self):
    """
    Function to download global weights from the server.
    """
    global_weights_path = "downloaded_global_weights.pth"
    if fedclient.download_global_weights():
        print(f"Downloaded global weights to {global_weights_path}")
        return global_weights_path
    else:
        print("Failed to download global weights.")
        return None

def calculate_accuracy(model, dataset, iteration, client_id):
    """
    Function to calculate the model's accuracy.
    """
    local_eval = fedclient.evaluate_model(model, dataset, iteration, client_id)
    accuracy = local_eval.get('accuracy', 0)
    return accuracy

def save_training_results_to_csv(client_id, iterations, accuracy_list):
    """
    Function to save training results to a CSV file.
    """
    csv_filename = f"client_{client_id}_training_results.csv"
    with open(csv_filename, 'w') as f:
        f.write("iterations,accuracy\n")
        for i, accuracy in zip(iterations, accuracy_list):
            f.write(f"{i},{accuracy}\n")
    print(f"Saved training results to {csv_filename}")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--upload', type=str, help="Upload local weights to server")
    parser.add_argument('--global-weights', type=str, help="Path to global weights file")
    parser.add_argument('--client-id', type=int, default=1, help="Client ID")

    args = parser.parse_args()
    global fedclient
    fedclient = FedClient()

    models = [YOLO('yolov8n.pt') for _ in range(2)]
    datasets = [
        "/Users/kuangsin/FedAvg/clients/client2/horizon/data.yaml",
        "/Users/kuangsin/FedAvg/clients/client2/top/data.yaml",
    ]

    accuracy_trend = []
    if args.train:
        for iteration in range(1, 6):  # Assuming 5 iterations
            print(f"Iteration {iteration}: Training models")
            local_weights_files = train_model(models, datasets, epochs, batch_size, args.client_id)
            
            print("Uploading local weights")
            if not upload_local_weights(local_weights_files, args.client_id):
                break
            
            print("Downloading global weights")
            global_weights_path = download_global_weights()
            if global_weights_path:
                global_weights = torch.load(global_weights_path)
                iteration_accuracies = []
                for model in models:
                    model.load_state_dict(global_weights)
                    accuracy = calculate_accuracy(model, datasets[0], iteration, args.client_id)
                    iteration_accuracies.append(accuracy)0
                    print(f"Client {args.client_id} Local evaluation accuracy: {accuracy}")
                
                if iteration_accuracies:
                    accuracy_trend.append(sum(iteration_accuracies) / len(iteration_accuracies))
            else:
                print("Global weights download failed, skipping accuracy calculation.")
        
        if accuracy_trend:
            save_training_results_to_csv(args.client_id, range(1, 6), accuracy_trend)
    if args.upload:
        print("Uploading local weights")
        if not upload_local_weights(args.upload , args.client_id):
            pass

if __name__ == "__main__":
    main()
