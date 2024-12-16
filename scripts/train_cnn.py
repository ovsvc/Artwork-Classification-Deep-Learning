#Heere we train custom CNN without using pretrained models and any optimization -> baseline
import sys
sys.path.append('/Users/viktoriiaovsianik/Documents/Uni/04_WS2024/06_ADL/Code/ADL-WS-2024')

import torch
import torchvision.transforms as transforms
from pathlib import Path
from scripts.metrics import Accuracy

from trainers.trainer import ImgClassificationTrainer
from datasets.dataset import Subset
from models.cnn import CNN_Net
from datasets.preprocessing import CustomDatasetPreprocessor
from datasets.AIArtBench import AIArtbench

# Add a debug flag to print statements at various stages
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

def prepare_datasets(dataset_path, debug_mode, fraction=1):

    debug_print("Preprocessing dataset...", debug_mode)
    preprocessor = CustomDatasetPreprocessor(dataset_path=dataset_path)
    preprocessor.preprocess(fraction=fraction)
    return preprocessor.get_splits(fraction=fraction)

def initialize_trainer(config, device):
    # Prepare datasets
    debug_mode = config['debug_mode']
    train_dataset, validation_dataset, _ = prepare_datasets(config['dataset_path'], debug_mode, config['fraction'])

    debug_print(f"Train dataset length: {len(train_dataset)}", debug_mode)
    debug_print(f"Validation dataset length: {len(validation_dataset)}", debug_mode)

    # Prepare the data for training
    debug_print("Preparing data transforms...", debug_mode)
    # Transforms
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Resize(size=(32, 32)),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])

    debug_print("Loading training and validation data...", debug_mode)
    train_data = AIArtbench(dataframe=train_dataset, subset=Subset.TRAINING, transform=train_transform)
    val_data = AIArtbench(dataframe=validation_dataset, subset=Subset.VALIDATION, transform=val_transform)

    debug_print(f"Check loaded data: {len(train_data)}", debug_mode)

    debug_print("Setting up the model and optimizer...", debug_mode)
    # Model, optimizer, and scheduler
    model = config['model']
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['scheduler_gamma'])

    # Debugging model setup
    debug_print(f"Model: {model}", debug_mode)

    # Loss and metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    train_metric = Accuracy(classes=train_dataset['label'].unique())
    val_metric = Accuracy(classes=validation_dataset['label'].unique())

    # Trainer
    model_save_dir = Path(config['model_save_dir'])
    model_save_dir.mkdir(exist_ok=True)

    return ImgClassificationTrainer(
        model=config['model'],
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=config['epochs'],
        training_save_dir=model_save_dir,
        batch_size=config['batch_size'],
        val_frequency=config['val_frequency'],
        debug_mode=config['debug_mode'],
        patience = config['patience']
    )

def train_model(config):
    model = config['model']
    device = get_device(config['debug_mode'])
    model = model.to(device)
    trainer = initialize_trainer(config, device)
    trainer.train()
    torch.save(trainer.model.state_dict(), Path(config['model_save_dir']) / "CNN_custom.pth")
    trainer.dispose()