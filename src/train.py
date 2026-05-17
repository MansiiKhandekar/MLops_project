import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess_data

def train_model():
    """
    Train a Random Forest model on the Diabetes dataset and save it.
    """
    # 1. Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 2. Initialize the model
    # We are using a Random Forest Classifier.
    # To demonstrate a source code change, you can modify 'n_estimators' or 'max_depth'
    # Example: change n_estimators=100 to n_estimators=10
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 3. Train the model on the training data
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # 4. Create the models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # 5. Save the trained model to a file
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()
