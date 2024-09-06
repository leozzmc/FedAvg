from flask import Flask, request, jsonify, send_file
from collections import OrderedDict
import torch
import os
import threading

app = Flask(__name__)

weights_dir = 'client_weights'
os.makedirs(weights_dir, exist_ok=True)
n = 2  # Number of clients
global_weights_file = 'global_weights.pth'
clients_notified = set()
uploaded_clients = set()
global_weights_updated = False

def average_weights(weights_list):
    """Averages the weights from multiple model state dictionaries (OrderedDict)."""
    average_weights = OrderedDict()
    n = len(weights_list)  # Number of model state dicts in the list
    for key in weights_list[0].keys():
        stacked_weights = torch.stack([weights[key].float() for weights in weights_list])
        average_weights[key] = torch.mean(stacked_weights, dim=0)
    return average_weights

def notify_clients():
    global clients_notified
    for client_id in range(1, n + 1):
        url = f"http://localhost:5000/api/notify_client/{client_id}"
        request.post(url)
    clients_notified = set()

@app.route('/api/upload_weights/<int:client_id>/<int:model_id>', methods=['POST'])
def upload_weights(client_id, model_id):
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
        for client_id, model_id in uploaded_clients:
            weights = torch.load(os.path.join(weights_dir, f'client_{client_id}_model_{model_id}_weights.pth'))
            weights_list.append(weights)
        global_weights = average_weights(weights_list)
        torch.save(global_weights, f'global_model_{model_id}_weights.pth')
        global_weights_updated = True
        uploaded_clients.clear()  # Clear the list for next iteration
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
