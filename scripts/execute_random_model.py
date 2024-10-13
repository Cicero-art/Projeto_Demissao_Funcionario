# Code for executing logistic regression model with random predictions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from evaluate_metrics import evaluate_model

# Load data
file_path = Path(__file__).parent.parent / "data" / "HR_comma_sep.csv"
df = pd.read_csv(file_path)

# Define target and features
y = df["left"]
X = df.drop(columns=["left"])

# Split data
_, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Manual threshold for binary classification
manual_threshold = 0.5

# Generate random continuous predictions between 0 and 1 (y_pred_proba)
random_predictions = np.random.rand(len(y_test))

# Apply threshold to convert continuous predictions to binary predictions
predictions_manual = (random_predictions >= manual_threshold).astype(int)

# Evaluate the baseline model with random predictions
metrics = evaluate_model(predictions_manual, y_test, random_predictions)

# Print metrics
print(f"Accuracy: {metrics['accuracy']:.2}")
print(f"ROC AUC: {metrics['roc_auc']:.2}")
print(f"Precision: {metrics['precision']:.2}")
print(f"Recall: {metrics['recall']:.2}")
