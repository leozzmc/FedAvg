from flask import Flask, request, jsonify, send_file
from collections import OrderedDict
import torch
import os

app = Flask(__name__)

weights_dir = 'client_weights'
os.makedirs(weights_dir, exist_ok=True)
n = 2  # Number of clients
global_weights_file = 'global_weights.pth'

def average_weights(weights_list):
    """Averages the weights from multiple model state dictionaries (OrderedDict)."""
    average_weights = OrderedDict()
    n = len(weights_list)  # Number of model state dicts in the list
    for key in weights_list[0].keys():
        stacked_weights = torch.stack([weights[key].float() for weights in weights_list])
        average_weights[key] = torch.mean(stacked_weights, dim=0)
    return average_weights

@app.route('/api/upload_weights/<int:client_id>', methods=['POST'])
def upload_weights(client_id):
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    filepath = os.path.join(weights_dir, f'client_{client_id}_weights.pth')
    file.save(filepath)

    # Check if all clients have uploaded their weights
    uploaded_files = [f for f in os.listdir(weights_dir) if f.startswith('client_') and f.endswith('_weights.pth')]
    
    if len(uploaded_files) == n:
        weights_list = []
        for uploaded_file in uploaded_files:
            weights = torch.load(os.path.join(weights_dir, uploaded_file))
            weights_list.append(weights)
        global_weights = average_weights(weights_list)
        torch.save(global_weights, global_weights_file)
        # Clear the client weights files
        for uploaded_file in uploaded_files:
            os.remove(os.path.join(weights_dir, uploaded_file))
        return jsonify({"status": "success"})

    return jsonify({"status": "pending"})

@app.route('/api/download_global_weights', methods=['GET'])
def download_global_weights():
    if not os.path.exists(global_weights_file):
        return jsonify({"status": "error", "message": "Global weights not available"}), 400

    # Return the global weights file
    return send_file(global_weights_file, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
