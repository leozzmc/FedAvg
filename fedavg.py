from ultralytics import YOLO
from collections import OrderedDict
import ipdb
import os
import copy


n = 2  # Number of clients
epochs_client = 50
epochs_global = 1
imgsz = 640  # Image size for training
root = '/tmp2/b11902010/YJ'
datasets = [root + '/Dataset/raw/machine1/classify/client1',
            root + '/Dataset/raw/machine1/classify/client2',
            root + '/Dataset/raw/machine1/classify/val']

models = [YOLO('yolov8n-cls.pt') for _ in range(n)]  

def get_files_in_path(directory_path) : 
    file_names = []
    # Loop through directory and get all files
    for entry in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, entry)):
            file_names.append(os.path.join(directory_path, entry))
    return file_names
def evaluate_model(model, dataset_path):
    results0 = model(get_files_in_path(dataset_path + '/0/'))
    results1 = model(get_files_in_path(dataset_path + '/1/'))
    cnt = 0
    tot = len(results0)
    for result in results0 : 
        cnt += (result.probs.data[0].item() > 0.5)
    tot += len(results1)
    for result in results1 : 
        cnt += (result.probs.data[1].item() > 0.5)
    return {"accuracy": cnt / tot}  # Example metric
# Function to simulate training on each client
def train_on_client(model, dataset_path, epochs):
    model.train(data=dataset_path, epochs=epochs, save=False, )
    return model.state_dict()  # Get the trained model weights


# Average the weights
import torch
def average_weights(weights_list):
    """Averages the weights from multiple model state dictionaries (OrderedDict)."""
    average_weights = OrderedDict()
    n = len(weights_list)  # Number of model state dicts in the list
    # Iterate over all weight keys in the first state dict in the list
    for key in weights_list[0].keys():
        # Stack the same weights from all state dicts and take the mean
        stacked_weights = torch.stack([weights[key].float() for weights in weights_list])
        average_weights[key] = torch.mean(stacked_weights, dim=0)
    return average_weights


results = []
# Main FedAvg loop
previous_weights = None
for epoch in range(epochs_global):
    client_weights = []
    epoch_results = {"local": [], "global": [], "validation" : []}
    for i in range(n):
        # Train locally
        local_weights = train_on_client(models[i], datasets[i], epochs_client)  # Train for 1 epoch at a time
        test_model = YOLO('last.pt')
        test_model.load_state_dict(local_weights)
        local_eval = evaluate_model(test_model, datasets[i]+'/test')
        print(i, local_eval)
        if epoch != 0 and local_eval['accuracy'] < results[-1]["local"][i]['accuracy'] : 
            epoch_results["local"].append(results[-1]["local"][i])
            local_weights = previous_weights[i]
        else : 
            epoch_results["local"].append(local_eval)

        
        # Collect weights for averaging
        client_weights.append(local_weights)
    
    # Average weights and update global model

    previous_weights = client_weights
    global_weights = average_weights(client_weights)
    test_model = YOLO('last.pt')
    test_model.load_state_dict(global_weights)
    validation_results = evaluate_model(test_model, datasets[-1])
    print('current epoch')
    for i in range(n):
        print(i)
        models[i].load_state_dict(global_weights)
        global_eval = evaluate_model(test_model, datasets[i]+'/test')
        epoch_results["global"].append(global_eval)

    # Store results for this epoch
    epoch_results["validation"].append(validation_results)
    print(epoch_results)
    results.append(epoch_results)

# Print or process results as needed
print(results)
