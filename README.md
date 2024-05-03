# Sanskrit OCR using Convolutional Neural Networks (CNN)

This project implements an OCR system for recognizing Sanskrit characters from images using Convolutional Neural Networks (CNN).

## Overview

This project utilizes deep learning techniques, specifically CNN architecture, to recognize Sanskrit characters from images. The CNN model is trained on a dataset of Devanagari Script Characters, comprising over 92,000 images of characters from the Devanagari script. The dataset includes characters from "ka" to "gya" consonants, digits 0 to 9, and is available in a CSV format. The images are 32x32 pixels in size and are represented as grayscale images.

## Dataset Information

The dataset contains 92,000 images of Devanagari characters, each with dimensions of 32x32 pixels. The CSV file has dimensions 92000 * 1025, with 1024 input features representing pixel values in grayscale (ranging from 0 to 255). The column "character" represents the Devanagari Character Name corresponding to each image.

### Acknowledgements

This dataset was originally created by the Computer Vision Research Group, Nepal. You can find the dataset on [Kaggle](https://www.kaggle.com/datasets/rishianand/devanagari-character-set).

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/sanskrit-OCR-CNN.git
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   
### Usage
Ensure you have the dataset (data.csv) containing Sanskrit character images. You can download it from Kaggle.
Run the sanskrit_ocr_cnn.ipynb notebook using Jupyter Notebook or any compatible environment.
Follow the instructions provided in the notebook to train the CNN model and perform OCR on Sanskrit character images.
Model Architecture
The CNN model architecture consists of convolutional layers, max-pooling layers, dropout layers, and fully connected layers. The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

### Results
After training, the model's performance can be evaluated using various metrics such as accuracy, loss, and classification reports. Additionally, visualizations such as training and validation accuracy/loss plots provide insights into the training process.


### Usage
Ensure you have the dataset (data.csv) containing Sanskrit character images. You can download it from Kaggle.
Run the sanskrit_ocr_cnn.ipynb notebook using Jupyter Notebook or any compatible environment.
Follow the instructions provided in the notebook to train the CNN model and perform OCR on Sanskrit character images.
Model Architecture
The CNN model architecture consists of convolutional layers, max-pooling layers, dropout layers, and fully connected layers. The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

### Results
After training, the model's performance can be evaluated using various metrics such as accuracy, loss, and classification reports. Additionally, visualizations such as training and validation accuracy/loss plots provide insights into the training process.

