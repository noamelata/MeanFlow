
import torch

def find_model(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return checkpoint


