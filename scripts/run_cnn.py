#Heere we train custom CNN without using pretrained models and any optimization -> baseline
import sys
sys.path.append('/Users/viktoriiaovsianik/Documents/Uni/04_WS2024/06_ADL/Code/ADL-WS-2024')

import torch
import torchvision.transforms as transforms
from pathlib import Path
from scripts.metrics import Accuracy

from trainers.ImgClassification import ImgClassification
from datasets.dataset import Subset
from datasets.preprocessing import CustomDatasetPreprocessor
from datasets.AIArtBench import AIArtbench


def debug_print(message, debug_mode):
    if debug_mode:
        print(message)

def get_device(debug_mode):
    # Checking device (CPU vs GPU)
    if torch.cuda.is_available():
        debug_print("CUDA (GPU) is available.", debug_mode)
    else:
        debug_print("CUDA (GPU) is not available. Training on CPU.", debug_mode)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_datasets(config):
    """
    Prepares the datasets for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary with dataset paths and preprocessing details.

    Returns:
        dict: Dictionary containing train, validation, and test datasets.
    """
    debug_mode = config['debug_mode']
    debug_print("Preprocessing dataset...", debug_mode)

    # Preprocess dataset and split
    preprocessor = CustomDatasetPreprocessor(dataset_path=config['dataset_path'])
    preprocessor.preprocess(fraction=config['fraction'])
    train_dataset, val_dataset, test_dataset = preprocessor.get_splits(fraction=config['fraction'])

    debug_print(f"Train dataset length: {len(train_dataset)}", debug_mode)
    debug_print(f"Validation dataset length: {len(val_dataset)}", debug_mode)
    debug_print(f"Test dataset length: {len(test_dataset)}", debug_mode)

    # Create datasets with transforms
    train_data = AIArtbench(dataframe=train_dataset, subset=Subset.TRAINING, transform=config['train_transform'])
    val_data = AIArtbench(dataframe=val_dataset, subset=Subset.VALIDATION, transform=config['val_transform'])
    test_data = AIArtbench(dataframe=test_dataset, subset=Subset.TEST, transform=config['test_transform'])

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_classes": train_dataset['label'].unique(),
        "val_classes": val_dataset['label'].unique(),
        "test_classes": test_dataset['label'].unique()
    }

def initialize_model_and_optimizer(config, classes, device):
    """
    Initializes the model, optimizer, scheduler, and loss function.

    Args:
        config (dict): Configuration dictionary with model and training settings.
        classes (list): Class labels for the dataset.
        device (torch.device): Device for training (CPU or GPU).

    Returns:
        dict: Dictionary containing the model, optimizer, scheduler, loss, and metrics.
    """
    debug_mode = config['debug_mode']
    model = config['model']
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['scheduler_gamma'])

    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Accuracy(classes=classes)

    debug_print(f"Model: {model}", debug_mode)
    
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "loss_fn": loss_fn,
        "metric": metric
    }

def initialize_trainer(config, datasets, model_components, device):
    """
    Initializes the trainer class with model, datasets, and training settings.

    Args:
        config (dict): Configuration dictionary with training parameters.
        datasets (dict): Dictionary of prepared datasets.
        model_components (dict): Initialized model, optimizer, scheduler, and metrics.
        device (torch.device): Device for training/testing.

    Returns:
        ImgClassification: Instance of the ImgClassification trainer.
    """
    return ImgClassification(
        model=model_components['model'],
        optimizer=model_components['optimizer'],
        loss_fn=model_components['loss_fn'],
        lr_scheduler=model_components['scheduler'],
        train_metric=model_components['metric'],
        val_metric=Accuracy(classes=datasets["val_classes"]),
        test_metric=Accuracy(classes=datasets["test_classes"]),
        train_data=datasets["train_data"],
        val_data=datasets["val_data"],
        test_data=datasets["test_data"],
        device=device,
        num_epochs=config['epochs'],
        training_save_dir=Path(config['model_save_dir']),
        batch_size=config['batch_size'],
        val_frequency=config['val_frequency'],
        debug_mode=config['debug_mode'],
        patience=config['patience']
    )

def train_model(config):
    """
    Trains the model using the ImgClassification trainer.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        ImgClassification: Trained ImgClassification trainer.
    """
    debug_mode = config['debug_mode']
    device = get_device(debug_mode)

    datasets = prepare_datasets(config)
    model_components = initialize_model_and_optimizer(config, classes=datasets["train_classes"], device=device)
    
    trainer = initialize_trainer(config, datasets, model_components, device)
    trainer.train()

    torch.save(trainer.model.state_dict(), Path(config['model_save_dir']) / f"{config['model_name']}.pth")
    trainer.dispose()

    return trainer

def test_model(config, trainer=None):
    """
    Tests the model on the test dataset.

    Args:
        config (dict): Configuration dictionary.
        trainer (ImgClassification, optional): Existing trainer object. If not provided, initializes a new one.

    Returns:
        None
    """
    debug_mode = config['debug_mode']
    device = get_device(debug_mode)

    # Use the existing trainer if provided, otherwise initialize a new one
    if not trainer:
        datasets = prepare_datasets(config)
        model_components = initialize_model_and_optimizer(config, classes=datasets["test_classes"], device=device)
        trainer = initialize_trainer(config, datasets, model_components, device)
        trainer.model.load_state_dict(torch.load(Path(config['model_save_dir']) / f"{config['model_name']}.pth", map_location=device))

    test_loss, test_accuracy, test_per_class_accuracy, all_labels, all_predictions, test_classes = trainer.test()
    return (test_loss, test_accuracy, 
            test_per_class_accuracy, all_labels, all_predictions, test_classes)
