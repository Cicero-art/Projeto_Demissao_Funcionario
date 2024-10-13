# Code for executing logistic regression model

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Generate random predictions
random_predictions = np.random.choice([0, 1], size=len(y_test))

# Evaluate the baseline model
metrics = evaluate_model(random_predictions, y_test)

# Print metrics
print(f"Accuracy: {metrics['accuracy']:.2}")
print(f"ROC AUC: {metrics['roc_auc']:.2}")
print(f"Precision: {metrics['precision']:.2}")
print(f"Recall: {metrics['recall']:.2}")
