import os
import PyTorch_Model
import torch



# Function to check total parameters
def check_total_params(model, max_params=20000):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")
    assert total_params <= max_params, f"Model exceeds maximum allowed parameters of {max_params}."


# Function to check BatchNorm usage
def check_batch_norm(model):
    has_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    assert has_batch_norm, "Model does not include any BatchNorm layers."
    print("BatchNorm check passed.")


# Function to check Dropout usage
def check_dropout(model):
    has_dropout = any(isinstance(layer, nn.Dropout) for layer in model.modules())
    assert has_dropout, "Model does not include any Dropout layers."
    print("Dropout check passed.")


# Function to check Fully Connected (Linear) layers
def check_fully_connected(model):
    has_fc = any(isinstance(layer, nn.Linear) for layer in model.modules())
    assert has_fc, "Model does not include any Fully Connected (Linear) layers."
    print("Fully Connected (Linear) check passed.")
