import torch

# load weights file
weights_path = 'global_weights.pth'
weights = torch.load(weights_path)

# Check weights
print(weights)
