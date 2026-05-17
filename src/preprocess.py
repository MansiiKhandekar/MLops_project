import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(data_path="data/diabetes.csv"):
    """
    Load the dataset from the specified path.
    """
    # Check if file exists, if not, print a helpful message
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please make sure it exists.")
        
    df = pd.DataFrame()
    # Read the CSV file using pandas
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Split the dataset into features (X) and target (y),
    and then split into training and testing sets.
    """
    # The target column is named 'Outcome', everything else is features
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Split the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # If this script is run directly, simply test the functions
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data successfully loaded and split!")
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size: {X_test.shape[0]} rows")
