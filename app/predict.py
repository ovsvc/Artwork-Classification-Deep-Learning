import os, sys, requests
import re
import torch
from torchvision import transforms
import torch.onnx
from PIL import Image


# Define paths for all supporting files & dataset
# Check if the code is running in Colab
IN_COLAB = 'google.colab' in sys.modules

if not IN_COLAB:
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    project_root = os.getenv('PROJECT_ROOT_PATH')
else:
    # Set the project root path for Colab
    project_root = userdata.get("project_root_path")

# Check if the project root path is set correctly
if project_root is None:
    raise ValueError("PROJECT_ROOT_PATH environment variable is not set.")

# Add the project root path to the system path
sys.path.append(project_root)

from models.resnet18 import ResNet18FineTuned
from models.simple_cnn import Simple_CNN


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_model(model_url, save_path):
    """
    Download a model from a URL and save it to the specified local file path.
    Ensures that the downloaded content is valid.
    """
    print(model_url)
    response = requests.get(model_url)
    if response.status_code == 200:
        # Check if the response is a valid PyTorch model (simple content check)
       # if b'PYTORCH' in response.content:
            with open(save_path, 'wb') as f:
                f.write(response.content)
          #  print(f"Model downloaded successfully to {save_path}")
        #else:
         #   print("Downloaded file is not a valid PyTorch model.")
         #   raise Exception("Error: Downloaded file is not a valid PyTorch model.")
    else:
        print(f"Failed to download model from {model_url}")
        raise Exception(f"Error {response.status_code}: Unable to download model.")


# Load model's dict
def load_model(model_url, model_filename, type):
    """
    Load a model from the 'loaded_models' directory in the current working directory (WD).
    If the model is not present, download it first.
    """
    # Get the current working directory
    current_wd = os.getcwd()

    # Create 'loaded_models' directory if it doesn't exist
    model_dir = os.path.join(current_wd, 'loaded_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory

    # Set the full model path inside the 'loaded_models' directory
    model_path = os.path.join(model_dir, model_filename)

    # Check if the model file exists locally, if not, download it
    if not os.path.exists(model_path):
        print(f"Model not found locally, downloading from {model_url}")
        download_model(model_url, model_path)

    # Load the model using PyTorch
    state_dict = torch.load(model_path, map_location= device) 

    # Define model
    if type == "CNN":
        classif_model = Simple_CNN()
    else:
        classif_model = ResNet18FineTuned()
    
    # Remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    classif_model.load_state_dict(state_dict)
    classif_model.to(device)
    classif_model.eval()

    return classif_model

# perform Classification
def classify(classification_model, content_image, device='cpu'):
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
        output = classification_model(content_image).cpu()  # Detach from GPU and move to CPU

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
            