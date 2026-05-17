import os
import sys
import joblib
import pytest
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocess import load_data, preprocess_data

def test_data_loading():
    """Test if the dataset loads successfully."""
    # This should not raise an exception
    df = load_data('../data/diabetes.csv')
    assert not df.empty, "Dataset is empty!"
    assert len(df.columns) == 9, "Dataset should have 9 columns (8 features + 1 target)"

def test_model_exists():
    """Test if the trained model file exists."""
    model_path = '../models/model.pkl'
    assert os.path.exists(model_path), f"Model file not found at {model_path}. Did you run train.py?"

def test_model_prediction():
    """Test if the model can make a prediction on a sample input."""
    model_path = '../models/model.pkl'
    # Only run this test if the model exists
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # Sample input (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        # Reshaped to 2D array as expected by scikit-learn
        sample_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        
        prediction = model.predict(sample_input)
        
        # Check if the prediction is valid (Diabetes targets are 0 or 1)
        assert prediction[0] in [0, 1], f"Invalid prediction: {prediction[0]}"
    else:
        pytest.skip("Model not found. Skipping prediction test.")
