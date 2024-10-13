import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset (label encoding, one-hot encoding)."""
    df['salary'] = LabelEncoder().fit_transform(df['salary'])
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    return df

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2):
    """Split the data into train and test sets."""
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Apply MinMax scaling to the data."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

