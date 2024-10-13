from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from evaluate_metrics import evaluate_model

# Load data
file_path = Path(__file__).parent.parent / "data" / "HR_comma_sep.csv"
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

# Make predictions with probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Set the manual threshold
manual_threshold = 0.64

# Adjust predictions based on the threshold
predictions_manual = (y_pred_proba >= manual_threshold).astype(int)

# Evaluate performance with the manual threshold
metrics_manual = evaluate_model(predictions_manual, y_test, y_pred_proba)

# Print metrics
print(f"Threshold: {manual_threshold:.2f}")
print(f"Accuracy: {metrics_manual['accuracy']:.2f}")
print(f"ROC AUC: {metrics_manual['roc_auc']:.2f}")
print(f"Precision: {metrics_manual['precision']:.2f}")
print(f"Recall: {metrics_manual['recall']:.2f}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random model)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
