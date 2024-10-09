import torch
import requests

class FedClient:
    def __init__(self):
        self.server_url = "http://localhost:5000"
    
    def train_on_client(self, model, dataset_path, epochs, batch_size, client_id, model_id):
        """Client-side model training logic."""
        model.train(data=dataset_path, epochs=epochs, imgsz=640, batch=batch_size)
        weights_file = f'model/client_{client_id}/model_{model_id}/client_{client_id}_model_{model_id}_weights.pth'
        model.save(weights_file)
        return weights_file

    def upload_weights(self, client_id, model_id, weights_file):
        """Upload the trained weights to the server."""
        with open(weights_file, 'rb') as f:
            response = requests.post(f"{self.server_url}/upload_weights", files={'file': f}, data={'client_id': client_id, 'model_id': model_id})
        return response.json()

    def download_global_weights(self, client_id, model_id):
        """Download the global weights from the server."""
        response = requests.get(f"{self.server_url}/download_global_weights/{client_id}/{model_id}")
        if response.status_code == 200:
            global_weights_file = f'model/global_weight/model_{model_id}_global_weights.pth'
            with open(global_weights_file, 'wb') as f:
                f.write(response.content)
            return global_weights_file
        return None

    def evaluate_model_logic(self, model, dataset_path):
        """Evaluates the model on the dataset."""
        results = model.val(data=dataset_path, imgsz=640)
        metrics = results.metrics
        accuracy = metrics.map() 
        bounding_box_areas = []  
        
        return accuracy, bounding_box_areas

    def evaluate_model(self, model, dataset_path, iteration, client_id):
        """Evaluate the model and return accuracy."""
        accuracy, bounding_box_areas = self.evaluate_model_logic(model, dataset_path)
        print(f"Client {client_id} Iteration {iteration}: Accuracy (mAP@0.5:0.95) = {accuracy}")
        return {"accuracy": accuracy, "bounding_box_areas": bounding_box_areas}