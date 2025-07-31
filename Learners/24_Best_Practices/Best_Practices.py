# Best_Practices.py
# file: data_utils.py
import pandas as pd

def load_data(path):
    """Load CSV data into DataFrame"""
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"File {path} not found.")
        return None

def clean_data(df):
    """Simple cleaning: drop missing values"""
    return df.dropna()

# file: model_utils.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def split_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")
    return acc

# file: test_utils.py
import unittest
import pandas as pd
from data_utils import load_data, clean_data

class TestDataUtils(unittest.TestCase):
    def test_load_data(self):
        df = load_data("non_existent.csv")
        self.assertIsNone(df)

    def test_clean_data(self):
        df = pd.DataFrame({"A": [1, 2, None], "B": [4, None, 6]})
        clean_df = clean_data(df)
        self.assertEqual(len(clean_df), 1)

if __name__ == "__main__":
    unittest.main()

# file: main.py
from data_utils import load_data, clean_data
from model_utils import split_data, train_model, evaluate_model

def main():
    df = load_data("sample_data.csv")
    if df is None:
        return
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target_col='target')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
