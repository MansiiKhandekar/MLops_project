import joblib
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess_data

def evaluate_model():
    """
    Load the trained model and evaluate it on the test dataset.
    Saves the accuracy score to metrics.txt.
    """
    # 1. Load and preprocess the data
    df = load_data()
    _, X_test, _, y_test = preprocess_data(df)
    
    # 2. Load the trained model
    model_path = 'models/model.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
    
    # 3. Make predictions on the test set
    predictions = model.predict(X_test)
    
    # 4. Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # 5. Save the metrics to a file
    with open('metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
    print("Metrics saved to metrics.txt")

if __name__ == "__main__":
    evaluate_model()
