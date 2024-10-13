import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score


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

