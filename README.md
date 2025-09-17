Markdown

# USD Bill Classification

A machine learning project that classifies US dollar bills based on images. The project uses a Convolutional Neural Network (CNN) to accurately identify bill denominations.

## Team Members

| AC.NO | Name | Role | Contributions |
| :---: | :---: | :---: | :--- |
|    202174125  | [Ehab Abdul Karim Al-Qobati] | Data Analyst | data preprocessing, and organization. |
|   202174202   | [Musa Ali Al-Salmany] | ML Engineer | Model development, training, and performance evaluation. |
|   202173006  | [Abdulraheem Yaser Abdulraheem] |Lead Developer,UI/UX Developer| Initial project setup, Building the Gradio user interface for the final application. |

## Installation and Setup

### Prerequisites

- Python 3.12.4 
- UV package manager
- **Note:** You must manually download the `usd-bill-classification-dataset.zip` file from Kaggle"https://www.kaggle.com/datasets/aishwaryatechie/usd-bill-classification-dataset/data" and place it in the `data/` folder.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/alhoripy/USD-Bill-Classification.git](https://github.com/alhoripy/USD-Bill-Classification.git)
    cd USD-Bill-Classification
    ```

2.  **Install dependencies using UV:**
    ```bash
    uv sync
    ```

## Project Structure

```bash
USD-Bill-Classification/
├── README.md               # Project documentation
├── pyproject.toml          # UV project configuration
├── .python-version         # Python version specification
├── test_project.py         # Test suite
├── .gitignore              # Files ignored by Git
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   │   ├── prepare_data.py # Script for data organization
│   ├── model/              # ML model implementations
│   │   ├── train_model.py  # Script for training the CNN model
│   └── ui/                 # User interface modules
│       └── app.py          # The Gradio web application
├── notebooks/              # Jupyter notebooks for analysis
│   ├── eda.ipynb           # Exploratory data analysis
│   └── model_training_report.ipynb # Model performance report
├── data/                   # Dataset files (ignored by Git)
│   ├── usd-bill-classification-dataset.zip
│   └── processed/          # Organized data for model training
├── models/                 # Saved trained models (ignored by Git)
└── docs/                   # Additional documentation
  └── model_performance.md


Usage ##

1. Data Preparation
First, you must process the raw data to prepare it for model training.

Bash

uv run python src/data/prepare_data.py
2. Model Training
Next, run the script to train the CNN model and save it to the models/ folder.

Bash

uv run python src/model/train_model.py
3. Launching the Application
Finally, launch the Gradio user interface to use the model.

Bash

uv run python src/ui/app.py
After running this command, open the URL provided in your terminal to access the web application.

 Results ##

Model Algorithm: Convolutional Neural Network (CNN)

Final Accuracy: [98.80]%

Key Findings: The CNN model effectively classifies different US dollar bill denominations with high accuracy. 
              The Confusion Matrix in the notebooks/ folder demonstrates the model's low rate of misclassification.

Contributing ##

Fork the repository

Create a feature branch: git checkout -b your-feature-name

Make your changes

Commit changes: git commit -m 'feat: Add your feature'

Push to branch: git push origin your-feature-name

Submit a pull request
