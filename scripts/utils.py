import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """Utility function to load a CSV file."""
    return pd.read_csv(file_path)
