import torch
import torch.nn as nn
from pathlib import Path


#custom imports
import os

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        if suffix:
            filename = f"model_{suffix}.pth"
        else:
            filename = "model.pth"
            
        save_path = os.path.join(save_dir, filename)

        torch.save(self.net.state_dict(), save_path)    
        print(f"Model saved at: {save_path}")


    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
 
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        try:
            self.load_state_dict(torch.load(path))
            print(f"Model loaded from: {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")