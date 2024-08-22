# Code for generating random predictions and evaluating them

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from evaluate_metrics import evaluate_model

# Load data
file_path = Path(__file__).parent.parent/"data"/"HR_comma_sep.csv"
df = pd.read_csv(file_path)


# Label encode 'salary'
df['salary'] = LabelEncoder().fit_transform(df['salary'])

# One-hot encode 'department'
df = pd.get_dummies(df, columns=['department'], drop_first=True)

# Define the target and features
y = df["left"]
X = df.drop(columns=["left"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate metrics
metrics = evaluate_model(predictions, y_test)

# Print metrics
print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"ROC AUC: {metrics['roc_auc']:.2}")
print(f"Precision: {metrics['precision']:.2}")
print(f"Recall: {metrics['recall']:.2}")
