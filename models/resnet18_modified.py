import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


class ResNet18Modified(nn.Module):
    def __init__(self, freeze_layers=False, num_classes=20, specific_layers_to_freeze=None):
        super(ResNet18Modified, self).__init__()
        
        # Load the pretrained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)

        self.resnet18.fc = nn.Sequential(
            nn.Dropout(p=0.25),  # Dropout with 50% probability
            nn.Linear(in_features=512, out_features=num_classes)
        )

        # If we want to freeze layers, freeze all by default and unfreeze specified layers
        if freeze_layers:
            self.freeze_all_layers()
            if specific_layers_to_freeze is not None:
                self.unfreeze_layers(specific_layers_to_freeze)


        self.get_trainable_params()
        
        self.name = "ResNet18_FineTuned"

    
    def freeze_all_layers(self):
        """Freeze all layers in the model."""
        for param in self.resnet18.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layers_to_freeze):
        """Unfreeze specific layers based on the provided list of layer names."""
        for name, layer in self.resnet18.named_children():
            # Check if the layer is not in the list of layers to unfreeze
            if name not in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = True
    

    def get_trainable_params(self):
        """Return a list of parameters that require gradients."""

        print("List of parameters that require gradients:")
        for name, param in self.resnet18.named_parameters():
            if param.requires_grad:
                print(name)


    def forward(self, x):
        return self.resnet18(x)

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
