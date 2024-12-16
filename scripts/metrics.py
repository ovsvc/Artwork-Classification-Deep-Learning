from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    Base class for a performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """
        pass


class Accuracy(PerformanceMeasure):
    """
    Measures overall and per-class accuracy for a classification task.
    """

    def __init__(self, classes: list[str]) -> None:
        """
        Initialize the accuracy object.
        
        :param classes: List of class names.
        """
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_predictions = 0
        self.total_predictions = 0
        self.class_correct = torch.zeros(len(self.classes), dtype=torch.float32)
        self.class_total = torch.zeros(len(self.classes), dtype=torch.float32)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the accuracy with new predictions and targets.
        
        :param prediction: Tensor of shape (s, c) where s is the number of samples
                           and c is the number of classes.
        :param target: Tensor of shape (s,) with true class labels.
        :raises ValueError: If shapes or values are invalid.
        """
        # Shape checks
        if prediction.ndim != 2 or target.ndim != 1:
            raise ValueError("Prediction must have shape (s, c) and target must have shape (s,).")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Prediction and target must have the same number of samples.")

        # Class range check
        num_classes = prediction.shape[1]
        if torch.any(target < 0) or torch.any(target >= num_classes):
            raise ValueError(f"Target values must be between 0 and {num_classes - 1}.")

        # Update overall statistics
        self.total_predictions += target.size(0)
        predicted_classes = torch.argmax(prediction, dim=1)
        correct = predicted_classes == target
        self.correct_predictions += correct.sum().item()

        # Update per-class statistics
        for i in range(target.size(0)):
            label = target[i].item()
            self.class_total[label] += 1
            if correct[i]:
                self.class_correct[label] += 1

    def accuracy(self) -> float:
        """
        Compute and return the overall accuracy.
        
        :return: Accuracy as a float between 0 and 1.
        """
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def per_class_accuracy(self) -> torch.Tensor:
        """
        Compute and return the per-class accuracy as a tensor.
        
        :return: Tensor containing per-class accuracy.
        """
        return torch.nan_to_num(self.class_correct / self.class_total, nan=0.0)

    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        
        :return: String summarizing accuracy and per-class accuracy.
        """
        overall_acc = self.accuracy()
        per_class_acc = self.per_class_accuracy()

        per_class_results = "\n".join(
            f"Accuracy for class {name}: {acc:.2f}"
            for name, acc in zip(self.classes, per_class_acc)
        )

        return f"Overall Accuracy: {overall_acc:.4f}\n Per-Class Accuracy:\n{per_class_results}"
    
    def get_classes(self):
        
        return self.classes








