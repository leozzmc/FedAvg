import os
import torch
import requests

class FedClient:
    
    def __init__(self) -> None:
        self.weights_file_template = 'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
        self.global_weights_file_template = 'model/client_{client_id}/model_{model_id}/downloaded_global_weights_model_{model_id}.pth'
    
    def evaluate_model_logic(self, model, dataset_path):
        # Load the dataset from the dataset_path (data.yaml) and perform evaluation
        results = model.val(data=dataset_path)  # `val` function is typically used for evaluation
        # Extract the accuracy or relevant metric
        accuracy = results.box.map  # Mean Average Precision (mAP)
        return accuracy
    
    def evaluate_model(self, model, dataset_path, iteration, client_id):
        accuracy = self.evaluate_model_logic(model, dataset_path)
        return {'accuracy': accuracy}

    def train_on_client(self, model, dataset_path, epochs, batch_size, client_id, model_id):
        model.train(data=dataset_path, epochs=epochs, batch=batch_size, save=True)
        local_weights = model.state_dict()  # Get the trained model weights
        # Save weights to file
        weights_file = self.weights_file_template.format(client_id=client_id, model_id=model_id)
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        torch.save(local_weights, weights_file)
        return weights_file

    def upload_weights(self, client_id, model_id, weights_file):
        url = f"http://localhost:5000/api/upload_weights/{client_id}/{model_id}"
        with open(weights_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)

        if response.status_code != 200:
            print(f"Failed to upload weights for client {client_id}, model {model_id}: {response.status_code}")
            print(response.text)
            return {"status": "error"}

        return response.json()

    # def download_global_weights(self, client_id, model_id):
    #     url = f"http://localhost:5000/api/download_global_weights/{client_id}/{model_id}"
    #     response = requests.get(url)

    #     if response.status_code == 200:
    #         global_weights_file = self.global_weights_file_template.format(client_id=client_id, model_id=model_id)
    #         os.makedirs(os.path.dirname(global_weights_file), exist_ok=True)
    #         with open(global_weights_file, 'wb') as f:
    #             f.write(response.content)
    #         return True
    #     else:
    #         print(f"Failed to download global weights for model {model_id}: {response.status_code}")
    #         print(response.text)
    #         return False
    def download_global_weights(self, client_id, model_id):
        url = f"http://localhost:5000/api/download_global_weights/{client_id}/{model_id}"
        response = requests.get(url)

        if response.status_code == 200:
            global_weights_file = self.global_weights_file_template.format(client_id=client_id, model_id=model_id)
            os.makedirs(os.path.dirname(global_weights_file), exist_ok=True)
            with open(global_weights_file, 'wb') as f:
                f.write(response.content)
            
        
            global_weights = torch.load(global_weights_file)
            return global_weights
        else:
            print(f"Failed to download global weights for model {model_id}: {response.status_code}")
            print(response.text)
            return None 


    def check_global_weights_status(self):
        url = "http://localhost:5000/api/check_global_weights_status"
        response = requests.get(url)
        return response.json()
