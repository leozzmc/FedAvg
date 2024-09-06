import os
import torch
import requests
import time

class FedClient:
    
    def __init__(self) -> None:
        self.weights_file_template = 'client_{client_id}_weights.pth'
        self.global_weights_file = 'downloaded_global_weights.pth'
    
    def evaluate_model_logic(self, model, dataset_path):
        # Load the dataset from the dataset_path (data.yaml) and perform evaluation
        results = model.val(data=dataset_path)  # `val` function is typically used for evaluation
        # Extract the accuracy or relevant metric
        accuracy = results.box.map  # Mean Average Precision (mAP)
        return accuracy

    def evaluate_model(self, model, dataset_path, iteration, client_id):
        accuracy = self.evaluate_model_logic(model, dataset_path)
        return {'accuracy': accuracy}

    # def train_on_client(self, model, dataset_path, epochs, batch_size, client_id):
    #     model.train(data=dataset_path, epochs=epochs, batch=batch_size, save=True, save_period=1, )
    #     local_weights = model.state_dict()  # Get the trained model weights
    #     # Save weights to file
    #     weights_file = self.weights_file_template.format(client_id=client_id)
    #     torch.save(local_weights, weights_file)
    #     return weights_file

    def train_on_client(self, model, dataset_path, epochs, batch_size, client_id):
        # Train the model and specify to save best model weights
        model.train(data=dataset_path, epochs=epochs, batch=batch_size, save=True, save_period=1)
        
        # YOLO saves the weights as `best.pt` automatically in its output folder
        output_folder = os.path.join('runs', 'train', 'exp', 'weights')
        best_weights_path = os.path.join(output_folder, 'best.pt')
        
        # Check if the best weights exist and rename to save with client id
        if os.path.exists(best_weights_path):
            # Rename the best.pt to include client ID for distinction
            new_weights_file = self.weights_file_template.format(client_id=client_id)
            os.rename(best_weights_path, new_weights_file)
            print(f"Best weights saved as {new_weights_file}.")
        else:
            print(f"Warning: Best weights file not found at {best_weights_path}.")

        return new_weights_file
    def upload_weights(self, client_id, weights_file):
        url = f"http://localhost:5000/api/upload_weights/{client_id}"
        with open(weights_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)

        if response.status_code != 200:
            print(f"Failed to upload weights for client {client_id}: {response.status_code}")
            print(response.text)
            return {"status": "error"}

        return response.json()

    def download_global_weights(self,  client_id):
        url = f"http://localhost:5000/api/download_global_weights/{client_id}"
        response = requests.get(url)

        if response.status_code == 200:
            with open(self.global_weights_file, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download global weights: {response.status_code}")
            print(response.text)
            return False
    
    # def check_server_for_global_weights(self, max_retry=10, wait_time=20):
    #     for _ in range(max_retry):
    #         response = requests.get("http://localhost:5000/api/download_global_weights")
    #         if response.status_code == 200:
    #             return True
    #         else:
    #             print("Global weights not available yet. Retrying...")
    #             time.sleep(wait_time)
    #     return False
    
    def check_global_weights_status(self):
        url = "http://localhost:5000/api/check_global_weights_status"
        response = requests.get(url)
        return response.json()

