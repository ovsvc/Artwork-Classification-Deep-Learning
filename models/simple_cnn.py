from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch


class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()
        
        # First Convolutional Layer: 32 filters, 5x5 kernel, 2 pixels padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        
        # Max Pooling Layer: 2x2 kernel, stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer: 16 filters, 3x3 kernel, 1 pixel padding
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        
        # Fully Connected Layer: 20 output units
        self.fc = nn.Linear(in_features=16 * 8 * 8, out_features=20)  # Flattened size is 16 * 8 * 8
        
        self.name = "Simple_CNN"

    def forward(self, x):
        # First Conv Layer + ReLU + MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second Conv Layer + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.view(-1, 16 * 8 * 8)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x

    def save(self, directory: Path, suffix: str = "best"):
        """
        Save the model state_dict to the specified directory.

        :param directory: The directory where the model will be saved.
        :param suffix: A suffix to be added to the model filename (default: "best").
        """
        # Ensure the directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # Define the file path
        model_path = directory / f"model_{suffix}.pth"

        # Save the model state_dict
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")
