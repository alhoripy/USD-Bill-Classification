import os
import zipfile
import shutil
import glob
import logging
import time

# Set up logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unzip_data(zip_path, extract_to):
    """Unzips a file to a specified directory."""
    if os.path.exists(extract_to):
        logging.info(f"Directory '{extract_to}' already exists. Skipping extraction.")
        return
    
    logging.info(f"Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Extraction complete.")
    except FileNotFoundError:
        logging.error(f"Error: {zip_path} not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during unzipping: {e}")
        raise

def organize_data(source_dir, dest_dir):
    """Organizes images into subfolders by their class/denomination."""
    if os.path.exists(dest_dir):
        logging.info(f"Destination directory '{dest_dir}' already exists. Removing it for a clean start.")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    logging.info("Organizing images into class folders...")
    
    # Use os.walk to find all files recursively
    found_images = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check for common image extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                
                # Extract class name from the folder name
                class_name = os.path.basename(os.path.dirname(source_path))
                
                # Create destination subfolder and copy the file
                class_dest_dir = os.path.join(dest_dir, class_name)
                os.makedirs(class_dest_dir, exist_ok=True)
                
                shutil.copy(source_path, class_dest_dir)
                found_images += 1
                
    if found_images > 0:
        logging.info(f"Successfully organized {found_images} images.")
    else:
        logging.warning("No images found in the extracted directory. Please check the folder structure.")
        
    logging.info("Data organization complete.")

def main():
    """Main function to prepare the dataset."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_folder = os.path.join(project_root, 'data')
    zip_file = os.path.join(data_folder, 'usd-bill-classification-dataset.zip')
    
    # Check if the zip file exists
    if not os.path.exists(zip_file):
        logging.error("Error: The dataset zip file was not found. Please download 'usd-bill-classification-dataset.zip' and place it in the 'data/' folder.")
        return

    # Unzip the raw data
    raw_data_path = os.path.join(data_folder, 'USD_Bill_Classification')
    unzip_data(zip_file, raw_data_path)
    
    # Add a small delay to ensure the OS has released file locks
    logging.info("Pausing for 2 seconds to ensure files are ready...")
    time.sleep(2)
    
    # Organize the data into a clean structure
    processed_data_path = os.path.join(data_folder, 'processed')
    organize_data(raw_data_path, processed_data_path)
    
    logging.info("Dataset preparation complete! The processed data is now in the 'data/processed' directory.")

if __name__ == "__main__":
    main()