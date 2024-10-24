from flask import Flask, jsonify, send_file, request 
from collections import OrderedDict
import torch
import os
import requests
import threading

app = Flask(__name__)

weights_dir = 'client_weights'
os.makedirs(weights_dir, exist_ok=True)
n = 2  # Number of clients
global_weights_file = 'global_weights.pth'
clients_notified = set()
uploaded_clients = set()
global_weights_updated = False

# Mock client data sizes for weighted averaging (e.g., data sizes per client)
client_data_sizes = {1: 1000, 2: 1500}  # Example: Client 1 has 1000 samples, Client 2 has 1500 samples

def weighted_average_weights(weights_list, weight_factors):
    """
    Averages the weights from multiple model state dictionaries (OrderedDict)
    using weighted averaging.

    Args:
    - weights_list (list): A list of model state dictionaries (OrderedDict).
    - weight_factors (list): A list of floats representing the weights for each model.
                             The length of weight_factors should match the length of weights_list.

    Returns:
    - average_weights (OrderedDict): The weighted average of the model weights.
    """
    average_weights = OrderedDict()
    total_weight = sum(weight_factors)  # Total sum of weights for normalization

    for key in weights_list[0].keys():
        # Stack the weights for this layer from all models and apply weighted average
        stacked_weights = torch.stack([weights[key].float() * weight_factors[i] for i, weights in enumerate(weights_list)])
        average_weights[key] = torch.sum(stacked_weights, dim=0) / total_weight

    return average_weights

def notify_clients():
    global clients_notified
    for client_id in range(1, n + 1):
        url = f"http://localhost:5000/api/notify_client/{client_id}"
        requests.post(url)  # Notifies the clients that global weights are ready
    clients_notified = set()

@app.route('/api/upload_weights/<int:client_id>/<int:model_id>', methods=['POST'])
def upload_weights(client_id, model_id):
    global global_weights_updated
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filepath = os.path.join(weights_dir, f'client_{client_id}_model_{model_id}_weights.pth')
    file.save(filepath)
    uploaded_clients.add((client_id, model_id))

    # Check if all clients have uploaded their weights for this model
    if len([c for c in uploaded_clients if c[1] == model_id]) == n:
        weights_list = []
        weight_factors = []
        for client_id, model_id in uploaded_clients:
            weights = torch.load(os.path.join(weights_dir, f'client_{client_id}_model_{model_id}_weights.pth'))
            weights_list.append(weights)
            weight_factors.append(client_data_sizes[client_id])
        
        # Calculate weighted average of the weights
        global_weights = weighted_average_weights(weights_list, weight_factors)
        torch.save(global_weights, f'global_model_{model_id}_weights.pth')
        global_weights_updated = True
        uploaded_clients.clear()  # Clear the list for next iteration
        notify_clients()  # Notifies all clients that global weights are ready
        return jsonify({"status": "success"})

    return jsonify({"status": "pending"})

@app.route('/api/download_global_weights/<int:client_id>/<int:model_id>', methods=['GET'])
def download_global_weights(client_id, model_id):
    global_weights_file = f'global_model_{model_id}_weights.pth'
    if not os.path.exists(global_weights_file):
        return jsonify({"status": "error", "message": "Global weights not available"}), 400

    return send_file(global_weights_file, as_attachment=True)

@app.route('/api/check_global_weights_status', methods=['GET'])
def check_global_weights_status():
    global global_weights_updated
    if global_weights_updated:
        return jsonify({"status": "updated"})
    else:
        return jsonify({"status": "not_updated"})

@app.route('/api/notify_client/<int:client_id>', methods=['POST'])
def notify_client(client_id):
    clients_notified.add(client_id)
    return jsonify({"status": "notified"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
