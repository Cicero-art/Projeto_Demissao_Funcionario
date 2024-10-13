from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_logistic_regression(X_train, y_train, penalty=None):
    """Train Logistic Regression model with given penalty."""
    model = LogisticRegression(penalty=penalty, solver='lbfgs', max_iter=1000) if penalty else LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def cross_validation(model, X_train, y_train):
    """Perform cross-validation and return accuracy, precision, recall, ROC AUC scores."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = {
        'accuracy': cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy').mean(),
        'roc_auc': cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc').mean(),
        'precision': cross_val_score(model, X_train, y_train, cv=skf, scoring='precision').mean(),
        'recall': cross_val_score(model, X_train, y_train, cv=skf, scoring='recall').mean()
    }
    return scores
