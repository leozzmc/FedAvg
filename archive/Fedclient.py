from ultralytics import YOLO
import os
import requests
import torch

class FedClient:
    
    def __init__(self) -> None:
        self.weights_file_template = 'client_{client_id}_weights.pth'
        self.global_weights_file = 'downloaded_global_weights.pth'

    @staticmethod
    def get_files_in_path(directory_path):
        file_names = []
        for entry in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, entry)):
                file_names.append(os.path.join(directory_path, entry))
        return file_names

    def evaluate_model(self, model, dataset_path):
        results0 = model(self.get_files_in_path(dataset_path))
        results1 = model(self.get_files_in_path(dataset_path))
        cnt = 0
        tot = len(results0)
        for result in results0:
            cnt += (result.probs.data[0].item() > 0.5)
        tot += len(results1)
        for result in results1:
            cnt += (result.probs.data[1].item() > 0.5)
        return {"accuracy": cnt / tot}  # Example metric

    def train_on_client(self, model, dataset_path, epochs, batch_size, client_id):
        model.train(data=dataset_path, epochs=epochs, batch=batch_size, save=True)
        local_weights = model.state_dict()  # Get the trained model weights
        # Save weights to file
        weights_file = self.weights_file_template.format(client_id=client_id)
        torch.save(local_weights, weights_file)
        return weights_file

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

    def download_global_weights(self):
        url = "http://localhost:5000/api/download_global_weights"
        response = requests.get(url)

        if response.status_code == 200:
            with open(self.global_weights_file, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download global weights: {response.status_code}")
            print(response.text)
            return False

# # Example usage
# if __name__ == "__main__":
#     fedclient = FedClient()
#     model = YOLO('yolov8n.pt')
#     # Example usage of FedClient methods
#     fedclient.train_on_client(model, '/path/to/data.yaml', epochs=10, batch_size=16, client_id=1)
