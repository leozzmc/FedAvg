from flask import Flask, request, jsonify, send_file
import torch
from ultralytics import YOLO
from collections import OrderedDict


app = Flask(__name__)

# Configure parameters
n = 2  # number of clients
epochs_global = 1  # global epoch
model_path = 'yolov8n-cls.pt'  # model path
global_model = YOLO(model_path)  # init global model


def average_weights(weights_list):
    """Averages the weights from multiple model state dictionaries (OrderedDict)."""
    average_weights = OrderedDict()
    n = len(weights_list)
    for key in weights_list[0].keys():
        stacked_weights = torch.stack([weights[key].float() for weights in weights_list])
        average_weights[key] = torch.mean(stacked_weights, dim=0)
    return average_weights

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    """Receive the model weight from the client to perform global aggregation"""
    file = request.files['file']
    local_weights = torch.load(file)
    
    #  client_weights list to store all weights from clients
    client_weights.append(local_weights)
    
    # If it is the last epoch, perform global aggregation
    if len(client_weights) == n:
        global_weights = average_weights(client_weights)
        global_model.load_state_dict(global_weights)
        return jsonify({"status": "weights aggregated and global model updated"})
    
    return jsonify({"status": "weights received"})

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """Return global model weights"""
    file_path = 'global_weights.pth'
    torch.save(global_model.state_dict(), file_path)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    client_weights = []
    app.run(host='0.0.0.0', port=5000)