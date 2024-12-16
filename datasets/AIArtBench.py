import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple
from datasets.dataset import Subset, Dataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

class AIArtbench(Dataset):
    """
    Custom Dataset class for loading images and labels from file paths
    provided in a preprocessed DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, subset: Subset, transform=None):
        """
        Initializes the dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame with 'filepath' and 'label' columns.
            subset (Subset): Subset type (TRAINING, VALIDATION, TEST).
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.dataframe = dataframe
        self.subset = subset
        self.transform = transform

        # Extract file paths and labels from the DataFrame
        self.filepaths = dataframe['filepath'].values
        self.labels = dataframe['label'].values

        # Extract class names from the labels
        self.classes = sorted(np.unique(self.labels))
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}

       # Map string labels to numeric indices
        self.label_indices = np.array([self.class_to_index[label] for label in self.labels])

        # Store data in a structure similar to 'self.dataset'
        self.dataset = {
            'data': [],
            'labels': self.label_indices,
        }
        # Check if files exist
        self._check_files_exist()

        # Load images into the dataset
        self._load_images()



    def _check_files_exist(self):
        """
        Check if all the files listed in the DataFrame exist.
        """
        for filepath in self.filepaths:
            if not os.path.isfile(filepath):
                raise ValueError(f"Image file not found: {filepath}")

    def _load_images(self):
        """
        Load images from file paths into the dataset and preprocess them.
        """
        for filepath in self.filepaths:
            image = self._load_image(filepath)
            # Convert image (numpy.ndarray) to torch.Tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
            self.dataset['data'].append(image_tensor)


    def _load_image(self, filepath: str) -> np.ndarray:
        """
        Loads an image from the given file path, ensures it is in RGB format,
        and converts it to a numpy array.

        Args:
            filepath (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image as a numpy array.
        """
        if not os.path.isfile(filepath):
            raise ValueError(f"Image file not found: {filepath}")

        try:
            with Image.open(filepath) as img:
                img = img.convert('RGB')  # Ensure image is in RGB format
                return np.array(img)
        except Exception as e:
            raise ValueError(f"Error loading image at {filepath}: {e}")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Returns the idx-th sample in the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[np.ndarray, int]: Tuple of (image, label_index).
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for the dataset.")

        # Get image and label index
        image = self.dataset['data'][idx]
        label_index = self.dataset['labels'][idx]

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        return image, label_index

    def num_classes(self) -> int:
        """
        Returns the number of unique classes in the dataset.
        """
        return len(self.classes)

    def plot_images(self, label_type: str = 'human', k: int = 5):
        """
        Plot k random images from either 'human' or 'AI' labels.

        Args:
            label_type (str): Either 'human' or 'AI'.
            k (int): Number of images to plot.
        """
        if label_type not in ['human', 'AI']:
            raise ValueError("label_type must be either 'human' or 'AI'.")
        
        # Filter the dataframe for the chosen label_type
        filtered_df = self.dataframe[self.dataframe['labels_type'] == label_type]
        
        if len(filtered_df) < k:
            raise ValueError(f"Not enough images available for label '{label_type}' to plot {k} images.")
        
        # Select k random samples
        sampled_df = filtered_df.sample(n=k, random_state=42)
        
        # Plot the images
        fig, axes = plt.subplots(1, k, figsize=(15, 5))
        fig.suptitle(f"{k} Random Images - {label_type.capitalize()}", fontsize=16)
        
        for i, (_, row) in enumerate(sampled_df.iterrows()):
            filepath = row['filepath']
            try:
                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(row['labels_category'], fontsize=10, pad=5)
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
        
        plt.tight_layout()
        plt.show()
