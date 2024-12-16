import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

def analyze_test_results(test_loss, test_accuracy, test_per_class_accuracy, all_labels, all_predictions, classes):
    """
    Analyzes the test results and calculates metrics such as confusion matrix, precision, recall, F1-score, 
    and accuracy. Plots the confusion matrix and prints a classification report.

    Args:
        test_loss (float): The test loss value.
        test_accuracy (float): The overall test accuracy.
        test_per_class_accuracy (list or np.ndarray): Per-class accuracy values.
        all_labels (np.ndarray): Ground-truth labels.
        all_predictions (np.ndarray): Model predictions.
        classes (list): Non-numerical names of the classes.

    Returns:
        dict: A dictionary containing precision, recall, F1-score, and accuracy.
    """

  
   # print(f"Per-Class Accuracy: {test_per_class_accuracy}")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate additional metrics
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))

    # Package metrics in a dictionary
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

    # Print summary
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    return metrics
