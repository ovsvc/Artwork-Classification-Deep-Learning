import os
import torch
import pandas as pd
from skimage import io
#from torch.utils.data import Dataset
from datasets.dataset import Subset, Dataset
from torchvision import transforms
from PIL import Image

class AIArtbench(Dataset):
    def __init__(self, dataframe, subset: Subset, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns 'filepath' and 'label'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.subset = subset

        # Extract class names and map them to indices
        self.class_names = sorted(self.dataframe['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)
  
    def get_classes(self):
        return self.class_names

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to fetch.

        Returns:
            Tuple (image, label): The image as a tensor and its corresponding label.
        """
        # Get image path from the dataframe
        img_name = self.dataframe.iloc[index, 0]  # Column 0: file path
        img_path = os.path.join(self.dataframe['filepath'][0], img_name)  # Assuming you have 'root_dir' column if needed
        
        # Load image using skimage or PIL
        image = io.imread(img_path)

        
        # Get the corresponding label
        label = self.dataframe.iloc[index, 3]  # Column 1: label
        
        # Convert the label into a numerical index
        label_idx = self.class_to_idx[label]

        # Convert the image to a PIL object for transformations (if any)
        image = Image.fromarray(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Convert the label to a tensor
        y_label = torch.tensor(label_idx, dtype=torch.long)

             # Print image size before transformation
       # print(f"Final image size: {image.shape}")

        return image, y_label
