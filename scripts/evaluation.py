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


def plot_confusion_matrix(all_labels, all_predictions, classes, normalize=False, fontsize=8):
    """
    Plots a tidier confusion matrix.

    Args:
        all_labels (np.ndarray): Ground-truth labels.
        all_predictions (np.ndarray): Model predictions.
        classes (list): Non-numerical names of the classes.
        normalize (bool): Whether to normalize the confusion matrix.
        fontsize (int): Font size for annotations.
    """
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size for clarity
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # Heatmap display
    plt.colorbar(cax)

    # Add labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation = 90, ha="right", fontsize=fontsize)
    ax.set_yticklabels(classes, fontsize=fontsize)
    
    # Annotate cells with values, omitting zeros for tidiness
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value != 0:  # Skip zero values
                ax.text(j, i, f"{value}", ha="center", va="center",
                        color="white" if value > cm.max() / 2 else "black", fontsize=fontsize)

    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel("Predicted Labels", fontsize=fontsize + 2)
    plt.ylabel("True Labels", fontsize=fontsize + 2)
    plt.tight_layout()
    plt.show()

    return cm





def analyze_test_results_v2(test_loss, test_accuracy, test_per_class_accuracy, all_labels, all_predictions, classes):
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
    cm = plot_confusion_matrix(all_labels, all_predictions, classes, normalize=True)

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
