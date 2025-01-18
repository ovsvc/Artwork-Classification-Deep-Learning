import os
import sys
import re
import torch
from torchvision import transforms
import torch.onnx
from PIL import Image
import streamlit as st
from models.resnet18 import ResNet18FineTuned

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get the project root path from environment variables
project_root = os.getenv('PROJECT_ROOT_PATH')

# Check if the environment variable is set correctly
if project_root is None:
    raise ValueError("PROJECT_ROOT_PATH environment variable is not set.")

# Add the project root path to the system path
sys.path.append(project_root)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#@st.cache
def load_model(model_path, map_location=device):
    print('load model')
    with torch.no_grad():
        classif_model = ResNet18FineTuned()
        state_dict = torch.load(model_path, map_location=device)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        classif_model.load_state_dict(state_dict)
        classif_model.to(device)
        classif_model.eval()
        return classif_model

#@st.cache
def stylize(style_model, content_image, device='cpu'):
    """
    Applies the style model to the content image and returns the stylized output.

    Args:
        style_model (torch.nn.Module): Pretrained style transfer model.
        content_image (str or PIL.Image.Image): Path to the content image or a PIL image.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The output image tensor from the style model.
    """
    # Load the image if a path is provided
    if isinstance(content_image, str):
        content_image = Image.open(content_image).convert('RGB')
    
    # Define the transformation pipeline
    content_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),  # Resize image
        transforms.ToTensor(),              # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Apply transformations
    content_image = content_transform(content_image).unsqueeze(0).to(device)  # Add batch dimension

    # Define the labels
    labels = [
        'AI: art nouveau',
        'AI: baroque',
        'AI: expressionism',
        'AI: impressionism',
        'AI: post impressionism',
        'AI: realism',
        'AI: renaissance',
        'AI: romanticism',
        'AI: surrealism',
        'AI: ukiyo-e',
        'Human: art nouveau',
        'Human: baroque',
        'Human: expressionism',
        'Human: impressionism',
        'Human: post impressionism',
        'Human: realism',
        'Human: renaissance',
        'Human: romanticism',
        'Human: surrealism',
        'Human: ukiyo-e'
    ]

    # Create class-to-index mapping
    class_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Create index-to-class mapping
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}


    # Perform inference with the model
    with torch.no_grad():
        output = style_model(content_image).cpu()  # Detach from GPU and move to CPU

        # Get the predicted class index
        predicted_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
        predicted_label = idx_to_class[predicted_idx]

        # Get the sorted scores for all classes
        softmax_output = torch.nn.functional.softmax(output, dim=1)  # Apply softmax to get probabilities
        sorted_scores, sorted_indices = torch.sort(softmax_output, descending=True)
        
        # Create a sorted list of class labels and their scores
        sorted_labels = [idx_to_class[idx.item()] for idx in sorted_indices[0]]
        # Convert the scores to integers (by rounding them or casting to int)
        sorted_scores_int = sorted_scores[0].cpu().numpy().astype(float)

    return predicted_label, list(zip(sorted_labels, sorted_scores_int))
            