import os
import pandas as pd
import random
from typing import Tuple


class CustomDatasetPreprocessor:
    """
    A class to preprocess a custom dataset with train, validation, and test splits.
    """
    def __init__(self, dataset_path: str):
        """
        Initializes the dataset preprocessor with the given dataset path.

        Args:
            dataset_path (str): Path to the dataset root directory.
        """
        self.dataset_path = dataset_path
        self.train_human = []
        self.train_ai = []
        self.test_human = []
        self.test_ai = []
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def _categorize_files(self):
        """
        Categorizes the files into human and AI directories for training and testing.
        """
        train_dir = os.path.join(self.dataset_path, 'train')
        test_dir = os.path.join(self.dataset_path, 'test')
        all_train_dir = os.listdir(train_dir)
        all_test_dir = os.listdir(test_dir)

        for dir in all_train_dir:
            if not dir.startswith('AI_'):
                self.train_human.append(os.path.join(train_dir, dir))
            else:
                self.train_ai.append(os.path.join(train_dir, dir))

        for dir in all_test_dir:
            if not dir.startswith('AI_'):
                self.test_human.append(os.path.join(test_dir, dir))
            else:
                self.test_ai.append(os.path.join(test_dir, dir))

    def _label_data(self, directories: list, label_type: str):
        """
        Labels files based on their directory structure.

        Args:
            directories (list): List of directories to process.
            label_type (str): Either 'human' or 'AI'.
        
        Returns:
            pd.DataFrame: Labeled dataset with file paths and categories.
        """
        filepaths, labels_type, labels_category = [], [], []

        for directory in directories:
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)
                if not os.path.isfile(filepath) or file.startswith("."):
                    continue
                category = os.path.basename(os.path.dirname(filepath))
                filepaths.append(filepath)
                labels_type.append(label_type)
                labels_category.append(category)

        # Clean up AI-specific prefixes
        cleaned_labels_category = [label.replace("AI_SD_", "").replace("AI_LD_", "") for label in labels_category]

        return pd.DataFrame({
            'filepath': filepaths,
            'labels_type': labels_type,
            'labels_category': cleaned_labels_category,
            'label': [f"{t}_{c}" for t, c in zip(labels_type, cleaned_labels_category)]
        })

    def _balance_data(self, data: pd.DataFrame, ai_sample_size: int = 5000):
        """
        Balances the dataset by sampling AI data and keeping human data as is.

        Args:
            data (pd.DataFrame): The dataset to balance.
            ai_sample_size (int): Number of samples per AI category.
        
        Returns:
            pd.DataFrame: Balanced dataset.
        """
        ai_data = data[data['label'].str.startswith('AI')]
        filtered_ai_data = []

        for label in ai_data['label'].unique():
            label_data = ai_data[ai_data['label'] == label]
            if len(label_data) > ai_sample_size:
                label_data = label_data.sample(n=ai_sample_size, random_state=42)
            filtered_ai_data.append(label_data)

        filtered_ai_data = pd.concat(filtered_ai_data)
        human_data = data[data['label'].str.startswith('human')]

        return pd.concat([filtered_ai_data, human_data])

    def _split_train_validation(self, data: pd.DataFrame, train_ratio: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and validation datasets.

        Args:
            data (pd.DataFrame): Dataset to split.
            train_ratio (float): Ratio of training data.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation datasets.
        """
        train_data, validation_data = [], []

        for label in data['label'].unique():
            label_data = data[data['label'] == label]
            label_data = label_data.sample(frac=1, random_state=42)
            train_split = int(len(label_data) * train_ratio)
            train_data.append(label_data.iloc[:train_split])
            validation_data.append(label_data.iloc[train_split:])

        return pd.concat(train_data).reset_index(drop=True), pd.concat(validation_data).reset_index(drop=True)

    def preprocess(self, fraction: float = 1.0):
        """
        Preprocesses the dataset: categorizes files, labels, balances, and splits.

        Args:
            fraction (float): The fraction of the dataset to use. Default is 1.0 (use all data).

        Returns:
            None
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError("Fraction must be between 0 and 1 (exclusive).")

        self._categorize_files()

        train_human_df = self._label_data(self.train_human, label_type="human")
        train_ai_df = self._label_data(self.train_ai, label_type="AI")
        all_train_data = pd.concat([train_human_df, train_ai_df], ignore_index=True)

        # Apply the fraction to training data
        if fraction < 1.0:
            all_train_data = all_train_data.sample(frac=fraction, random_state=42).reset_index(drop=True)

        balanced_train_data = self._balance_data(all_train_data)
        self.train_data, self.validation_data = self._split_train_validation(balanced_train_data)

        test_human_df = self._label_data(self.test_human, label_type="human")
        test_ai_df = self._label_data(self.test_ai, label_type="AI")
        self.test_data = pd.concat([test_human_df, test_ai_df], ignore_index=True)

        # Apply the fraction to test data
        if fraction < 1.0:
            self.test_data = self.test_data.sample(frac=fraction, random_state=42).reset_index(drop=True)

    def get_splits(self, fraction: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns the training, validation, and test datasets.

        Args:
            fraction (float): The fraction of the dataset to return. Default is 1.0 (use all data).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test datasets.
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError("Fraction must be between 0 and 1 (exclusive).")

        if self.train_data is None or self.validation_data is None or self.test_data is None:
            raise RuntimeError("Preprocessing has not been run. Call `preprocess()` first.")

        # Sample a fraction of the splits
        train_sample = self.train_data.sample(frac=fraction, random_state=42).reset_index(drop=True)
        validation_sample = self.validation_data.sample(frac=fraction, random_state=42).reset_index(drop=True)
        test_sample = self.test_data.sample(frac=fraction, random_state=42).reset_index(drop=True)

        return train_sample, validation_sample, test_sample