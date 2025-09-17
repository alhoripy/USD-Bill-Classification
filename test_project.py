"""
Test Project Script
Verify that all components of the USD Bill Classification project work correctly.
"""

import sys
import os
# Add project's root to Python path
# Add project's root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing module imports...")
    try:
        from src.data import prepare_data
        from src.model import train_model
        from src.ui import app
        print("✅ All project modules imported successfully.")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_directory_exists():
    """Test that the processed data directory exists."""
    print("\n📊 Testing data directory...")
    project_root = os.path.abspath(os.path.dirname(__file__))
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    
    if os.path.exists(processed_data_path) and len(os.listdir(processed_data_path)) > 0:
        print(f"✅ Processed data directory '{processed_data_path}' exists and is not empty.")
        return True
    else:
        print(f"❌ Processed data directory not found or is empty. Please run 'uv run python src/data/prepare_data.py'.")
        return False

def test_model_can_be_loaded():
    """Test that a trained model file can be loaded without errors."""
    print("\n🤖 Testing model loading...")
    project_root = os.path.abspath(os.path.dirname(__file__))
    models_path = os.path.join(project_root, 'models')
    
    try:
        list_of_models = glob.glob(os.path.join(models_path, '*.keras'))
        if not list_of_models:
            print(f"❌ No trained model file (.keras) found in '{models_path}'. Please run 'uv run python src/model/train_model.py'.")
            return False
        
        latest_model_path = max(list_of_models, key=os.path.getctime)
        model = load_model(latest_model_path)
        
        if model:
            print(f"✅ Model loaded successfully from: {latest_model_path}")
            return True
        else:
            print("❌ Failed to load the model.")
            return False
            
    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")
        return False

def test_model_prediction():
    """Test that the loaded model can make a prediction."""
    print("\n🎯 Testing model prediction...")
    try:
        # Load the model
        project_root = os.path.abspath(os.path.dirname(__file__))
        models_path = os.path.join(project_root, 'models')
        list_of_models = glob.glob(os.path.join(models_path, '*.keras'))
        latest_model_path = max(list_of_models, key=os.path.getctime)
        model = load_model(latest_model_path)
        
        # Create a dummy image for testing
        dummy_image = np.zeros((150, 150, 3), dtype=np.uint8)
        dummy_image = np.expand_dims(dummy_image, axis=0) # Add batch dimension
        
        # Test a prediction
        predictions = model.predict(dummy_image)
        
        if predictions.shape[1] == 12: # Assuming 6 classes (1, 2, 5, 10, 50, 100)
            print("✅ Model prediction output shape is correct.")
            return True
        else:
            print(f"❌ Model output shape is incorrect. Expected 12, got {predictions.shape[1]}.")
            return False

    except Exception as e:
        print(f"❌ An error occurred during model prediction: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 USD Bill Classification - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_directory_exists,
        test_model_can_be_loaded,
        test_model_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
            
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())