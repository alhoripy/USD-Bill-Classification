import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(input_shape, num_classes):
    """Creates a simple Convolutional Neural Network (CNN) model."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5), # Add dropout for regularization
        Dense(num_classes, activation='softmax') # Output layer
    ])
    return model

def main():
    """Main function to train and save the model."""
    logging.info("Starting model training process.")
    
    # Define file paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'processed')
    models_path = os.path.join(project_root, 'models')

    # Check if data exists
    if not os.path.exists(data_path):
        logging.error("Processed data folder not found. Please run `prepare_data.py` first.")
        return
    
    # Create the models directory if it doesn't exist
    os.makedirs(models_path, exist_ok=True)
    
    # Define image properties and training parameters
    image_size = (150, 150)
    batch_size = 32
    epochs = 10 # Start with a small number of epochs
    
    # Use ImageDataGenerator to load images from directories
    datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values
        validation_split=0.2 # Use 20% of data for validation
    )

    logging.info("Loading training and validation data...")
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    
    # Create and compile the model
    model = create_model(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    logging.info("Model compilation complete. Starting training...")
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Save the trained model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"usd_bill_classifier_{timestamp}.keras" # Note: Keras model format
    model_path = os.path.join(models_path, model_name)
    model.save(model_path)
    
    logging.info(f"Model saved successfully at: {model_path}")
    logging.info("Model training process finished.")

if __name__ == "__main__":
    main()