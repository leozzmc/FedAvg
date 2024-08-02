from ultralytics import YOLO
from collections import OrderedDict
import os
import torch

# Configuration parameters
n = 2  # Number of clients
epochs_client = 50  # Number of epochs for each client's training
epochs_global = 1  # Number of global aggregation rounds
imgsz = 640  # Image size for training
root = os.getcwd()  # Set root to the current working directory
print(root)
datasets = [
    os.path.join(root, 'client1_dataset'),  # Adjust these paths as needed
    os.path.join(root, 'client2_dataset'),
    os.path.join(root, 'validation_dataset')
]

# Initialize models (Replace 'yolov8n-cls.pt' with the path to the desired model)
model_paths = ['yolov8n-cls.pt' for _ in range(n)]  # Replace with your model paths
models = [YOLO(model_path) for model_path in model_paths]

def get_files_in_path(directory_path):
    """Retrieve all file paths in a given directory."""
    file_names = []
    for entry in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, entry)):
            file_names.append(os.path.join(directory_path, entry))
    # if the entry is a file, then return it's full path
    return file_names


def evaluate_model(model, dataset_path):
    """Evaluate model accuracy on a dataset."""
    results0 = model(get_files_in_path(os.path.join(dataset_path, '0')))
    results1 = model(get_files_in_path(os.path.join(dataset_path, '1')))
    cnt = 0     # Counter to calculate the number of data wit
    tot = len(results0) 
    for result in results0:
        cnt += (result.probs.data[0].item() > 0.5)
    tot += len(results1)
    for result in results1:
        cnt += (result.probs.data[1].item() > 0.5)
    return {"accuracy": cnt / tot}  # Return accuracy as a metric

def train_on_client(model, dataset_path, epochs):
    """Train the model on a client dataset."""
    model.train(data=dataset_path, epochs=epochs, save=False)
    return model.state_dict()  # Return the trained model weights

def average_weights(weights_list):
    """Averages the weights from multiple model state dictionaries (OrderedDict)."""
    average_weights = OrderedDict()
    n = len(weights_list)  # Number of model state dicts in the list
    for key in weights_list[0].keys():
        stacked_weights = torch.stack([weights[key].float() for weights in weights_list])
        average_weights[key] = torch.mean(stacked_weights, dim=0)
    return average_weights

# Main FedAvg loop
results = []
previous_weights = None
for epoch in range(epochs_global):
    client_weights = []
    epoch_results = {"local": [], "global": [], "validation": []}
    
    for i in range(n):
        # Train locally on client
        local_weights = train_on_client(models[i], datasets[i], epochs_client)
        
        # Evaluate locally trained model
        test_model = YOLO('last.pt')  # Replace with your model initialization
        test_model.load_state_dict(local_weights)
        local_eval = evaluate_model(test_model, os.path.join(datasets[i], 'test'))
        print(f"Client {i} local evaluation: {local_eval}")
        
        if epoch != 0 and local_eval['accuracy'] < results[-1]["local"][i]['accuracy']:
            epoch_results["local"].append(results[-1]["local"][i])
            local_weights = previous_weights[i]
        else:
            epoch_results["local"].append(local_eval)
        
        # Collect weights for averaging
        client_weights.append(local_weights)
    
    # Average weights and update global model
    previous_weights = client_weights
    global_weights = average_weights(client_weights)
    
    # Evaluate global model on validation dataset
    test_model = YOLO('last.pt')  # Replace with your model initialization
    test_model.load_state_dict(global_weights)
    validation_results = evaluate_model(test_model, datasets[-1])
    print('Validation results:', validation_results)
    
    # Evaluate global model on each client's test dataset
    for i in range(n):
        models[i].load_state_dict(global_weights)
        global_eval = evaluate_model(test_model, os.path.join(datasets[i], 'test'))
        epoch_results["global"].append(global_eval)
        print(f"Client {i} global evaluation: {global_eval}")

    # Store results for this epoch
    epoch_results["validation"].append(validation_results)
    results.append(epoch_results)

# Print or process results as needed
print(results)
