import torch, os
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys



# Check if the code is running in Colab
IN_COLAB = 'google.colab' in sys.modules

if not IN_COLAB:
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    project_root = os.getenv('PROJECT_ROOT_PATH')
    
else:
    from google.colab import userdata
    # Set the project root path for Colab
    project_root = userdata.get("project_root_path")

# Check if the project root path is set correctly
if project_root is None:
    raise ValueError("PROJECT_ROOT_PATH environment variable is not set.")

# Add the project root path to the system path
sys.path.append(project_root)

from scripts.wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass

class ImgClassification(BaseTrainer):
    """
    Class that stores the logic for training and testing a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        test_metric,
        train_data,
        val_data,
        test_data,
        device,
        num_epochs: int, 
        training_save_dir: Path,
        debug_mode = False,
        batch_size: int = 4,
        val_frequency: int = 5,
        patience: int = 3,

    ) -> None:
        """
        Initializes the trainer with model, data, metrics, etc.
        """

        self.model = model
        self.debug_mode = debug_mode
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.train_metric = train_metric
        self.val_frequency = val_frequency
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.batch_size = batch_size    
        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)
        self.num_test_data = len(test_data)
        self.training_save_dir = training_save_dir
        self.patience = patience

        #DataLoaders
        self.train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True,
            num_workers = 0)
        self.val_data_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            num_workers = 0)
        self.test_data_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False,
            num_workers = 0)

        #WanDB Logger
        self.wandb_logger = WandBLogger(enabled=True, model=model, run_name=model._get_name())
        
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        if self.debug_mode:
            print(f"--- Training epoch {epoch_idx} ---")


        self.model.train() 
        self.train_metric.reset()

        epoch_loss = 0


        for i, batch in tqdm(enumerate(self.train_data_loader), desc="Train", total=len(self.train_data_loader)):

            inputs, labels = batch
            batch_size = inputs.shape[0]

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.loss_fn(outputs, labels)

            # Backpropagate and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Gather metrics
            epoch_loss += (loss.item() * batch_size)
            self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())

            if self.debug_mode and i % 500 == 0:  # Print debug info every 10 batches
                print(f"Batch {i}, Loss: {loss.item()}")
      

        # Update learning rate scheduler
        self.lr_scheduler.step()
        epoch_loss /= self.num_train_data

        if self.debug_mode:
            print(f"Epoch {epoch_idx} Training Loss: {epoch_loss}")
            print(f"Training Metrics: {self.train_metric}")

        return epoch_loss, self.train_metric.accuracy(), self.train_metric.per_class_accuracy()

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """

        if self.debug_mode:
            print(f"--- Validating epoch {epoch_idx} ---")

        self.val_metric.reset()
        epoch_loss = 0.
        

        for i, batch in tqdm(enumerate(self.val_data_loader), desc="Evaluate", total=len(self.val_data_loader)):
            self.model.eval()
            with torch.no_grad():
                inputs, labels = batch
                batch_size = inputs.shape[0] 
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_fn(outputs, labels)

                # Update metrics
                epoch_loss += loss.item() * batch_size
                self.val_metric.update(outputs.cpu(), labels.cpu())

            if self.debug_mode and i % 500 == 0:  # Print debug info every 10 batches
                print(f"Batch {i}, Validation Loss: {loss.item()}")

        epoch_loss /= self.num_val_data

        if self.debug_mode:
            print(f"Epoch {epoch_idx} Validation Loss: {epoch_loss}")
            print(f"Validation Metrics: {self.val_metric}")

        return (
            epoch_loss,
            self.val_metric.accuracy(),
            self.val_metric.per_class_accuracy()
        )

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """

        best_accuracy = 0.0
        best_loss = np.inf
        early_stopping_counter = 0 

        print(f"Training with batch size: {self.batch_size}")

        for epoch_idx in range(self.num_epochs):
            print(f"Epoch {epoch_idx}/{self.num_epochs}:")

            train_loss, train_acc, train_acc_class = self._train_epoch(epoch_idx)

            wandb_log = {'epoch': epoch_idx, 'train/loss': train_loss, 'train/acc': train_acc}

            if epoch_idx % self.val_frequency == 0:
                val_loss, val_acc, val_acc_class = self._val_epoch(epoch_idx)
                wandb_log.update({"val/loss": val_loss, "val/acc": val_acc})


                if best_accuracy <= val_acc and best_loss >= val_loss:
                    print(f"#### Best accuracy {val_acc} at epoch {epoch_idx}")
                    print(f"#### Saving model to {self.training_save_dir}")
                    self.model.save(Path(self.training_save_dir), suffix="best")
                    best_accuracy = val_acc
                    best_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(f"Early stopping counter: {early_stopping_counter}/{self.patience}")


                if epoch_idx == self.num_epochs-1:
                    self.model.save(Path(self.training_save_dir), suffix="last")
                
                # Check if early stopping condition is met
                if early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            self.wandb_logger.log(wandb_log)

    def test(self) -> Tuple[float, float, float]:
        """
        Tests the model on a given test dataset.
        Prints the metrics and returns loss, accuracy, and per-class accuracy.

        Args:
            test_data (torch.utils.data.Dataset): Test dataset to evaluate on.
            batch_size (int): Batch size for the test DataLoader.

        Returns:
            Tuple[float, float, float]: Test loss, mean accuracy, and mean per-class accuracy.
        """

        print("Testing the model...")

        self.wandb_logger = WandBLogger(enabled=True, model=self.model, run_name=self.model._get_name())
        

        self.model.eval()  # Set model to evaluation mode

        print("Model name...", self.model._get_name())

        # Grad-CAM variables
        activations = []
        gradients = []

        if self.model._get_name() == "ResNet18FineTuned":
             # Define hooks to capture activations and gradients


            def forward_hook(module, input, output):
                """
                Capture the activations of the target layer.
                """
                activations.append(output)

            def backward_hook(module, grad_input, grad_output):
                """
                Capture the gradients of the target layer.
                """
                gradients.append(grad_output[0])

            # Register hooks for the target layer
            target_layer = self.model.resnet18.layer4[1].conv2
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
   

        test_loss = 0.0
        self.test_metric.reset()

        all_labels = []
        all_predictions = []

        #with torch.no_grad():
        for i, batch in tqdm(enumerate(self.test_data_loader), desc="Test", total=len(self.test_data_loader)):
            inputs, labels = batch
            batch_size = inputs.size(0)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)

            # Compute loss
            loss = self.loss_fn(outputs, labels)
            test_loss += loss.item() * batch_size

            # Backward pass for gradients (only if Grad-CAM hooks are active)
            if self.model._get_name() == "ResNet18FineTuned":
                self.model.zero_grad()  # Clear existing gradients
                loss.backward()  # Compute gradients

            # Update metrics
            self.test_metric.update(outputs.cpu(), labels.cpu())

            # Store original and predicted labels
            all_labels.extend(labels.cpu().numpy())  # Convert to numpy for easier comparison
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        # Compute average loss
        test_loss /= self.num_test_data

        # Gather final metrics
        test_accuracy = self.test_metric.accuracy()
        test_per_class_accuracy = self.test_metric.per_class_accuracy()

        # Convert labels and predictions to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        if self.debug_mode:
            print(f"Test Metrics: {self.test_metric}")
        # Log metrics to WandB
        self.wandb_logger.log({"test/loss": test_loss, "test/accuracy": test_accuracy})

        return test_loss, test_accuracy, test_per_class_accuracy, all_labels, all_predictions, self.test_metric.get_classes(), activations, gradients

    def dispose(self) -> None:
        """
        Finish logging.
        """
        self.wandb_logger.finish()
