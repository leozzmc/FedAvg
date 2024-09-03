import os
import matplotlib.pyplot as plt
import torch
import requests

class FedClient:
    
    def __init__(self) -> None:
        self.weights_file_template = 'client_{client_id}_weights.pth'
        self.global_weights_file = 'downloaded_global_weights.pth'
        self.accuracy_history = {}
    
    def plot_accuracy_trend(self, client_id, iteration, accuracy, save_dir='accuracy_trends'):
        # Initialize accuracy history for the client if not already present
        if client_id not in self.accuracy_history:
            self.accuracy_history[client_id] = []

        # Update accuracy history for the specific client
        self.accuracy_history[client_id].append((int(iteration), accuracy))

        # Plot the line graph
        plt.figure()
        iterations, accuracies = zip(*self.accuracy_history[client_id])
        plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b')
        plt.title(f'Global Accuracy Trend over FedAvg Iterations (Client {client_id})')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot
        save_path = os.path.join(save_dir, f'accuracy_trend_client_{client_id}.png')
        
        print(f"Saving plot to: {save_path}")  # Debug info
        
        plt.savefig(save_path)
        plt.close()
    
    
    def evaluate_model_logic(self, model, dataset_path):
        # Load the dataset from the dataset_path (data.yaml)
        # and perform evaluation
        results = model.val(data=dataset_path)  # `val` function is typically used for evaluation
        
        # Extract the accuracy or relevant metric
        accuracy = results.box.map  # Mean Average Precision (mAP)
        
        return {"accuracy": accuracy}

    def evaluate_model(self, model, dataset_path, iteration, client_id):
        accuracy_result = self.evaluate_model_logic(model, dataset_path)  # Pass the file directly
        accuracy = accuracy_result['accuracy']
        
        input("Please enter to continue...")  # Wait for user input before printing accuracy
        # Save and update accuracy trend
        self.plot_accuracy_trend(client_id=client_id, iteration=iteration, accuracy=accuracy)

        return accuracy_result

    def train_on_client(self, model, dataset_path, epochs, batch_size, client_id):
        model.train(data=dataset_path, epochs=epochs, batch=batch_size, save=False)
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
