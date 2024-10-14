import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, roc_curve


def evaluate_model(predictions: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
     Evaluates the performance of a model using various metrics.

     Parameters:
     - predictions: numpy array of model predictions
     - y_true: numpy array of true target values
     - y_pred_proba: numpy array of predicted probabilities for the positive class (1)

     Returns:
     - A dictionary containing the accuracy, ROC AUC, precision, and recall
    """
    metrics = {
        'accuracy': accuracy_score(y_true, predictions),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions)
    }
    return metrics

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def calculate_high_risk_employees(y_pred_proba, threshold: float):
    """Calculate percentage of employees at high risk based on threshold."""
    count_high_risk = sum(y_pred_proba >= threshold)
    percentage_high_risk = (count_high_risk / len(y_pred_proba)) * 100
    return count_high_risk, percentage_high_risk
