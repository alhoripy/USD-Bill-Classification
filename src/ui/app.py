import os
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import glob

# Define paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
models_path = os.path.join(project_root, 'models')

# Load the latest saved model
try:
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"The 'models' directory was not found at {models_path}. Please run model training first.")
    
    list_of_models = glob.glob(os.path.join(models_path, '*.keras'))
    
    if not list_of_models:
        raise FileNotFoundError("No trained model file (.keras) found inside the 'models' directory.")
        
    latest_model_path = max(list_of_models, key=os.path.getctime)
    model = load_model(latest_model_path)
    print(f"Model loaded successfully from: {latest_model_path}")
    
except FileNotFoundError as e:
    print(e)
    model = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    model = None

# Define class names based on the data directory
data_path = os.path.join(project_root, 'data', 'processed')
if os.path.exists(data_path):
    class_names = sorted(os.listdir(data_path))
else:
    class_names = ["1", "2", "5", "10", "20", "50", "100"]
    print("Warning: Data directory not found. Using default class names.")

def predict_bill(image):
    """Predicts the class of a USD bill from an input image."""
    if model is None:
        return "Model not available. Please train the model first.", None
    
    # Preprocess the image
    image = Image.fromarray(image.astype('uint8'), 'RGB').resize((150, 150))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension
    
    # Make a prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    
    # Return results with confidence scores
    confidences = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    return f"Predicted Class: ${predicted_class}", confidences
# هذا هو الكود الذي يجب أن يكون في نهاية ملف app.py

if __name__ == "__main__":
    gr.Interface(
        fn=predict_bill,
        inputs=gr.Image(type="numpy", label="Upload an image of a USD bill"),
        outputs=[gr.Textbox(label="Prediction"), gr.Label(label="Confidence")],
        title="USD Bill Classification",
        description="Upload an image of a USD bill to classify its denomination."
    ).launch()