# Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using CNN-AE

This project implements a Convolutional Autoencoder (CNN-AE) to denoise multispectral transmission images, which is crucial for early breast cancer screening. The model learns to reconstruct clean images from noisy multispectral images, improving image clarity for accurate analysis.

## Features
- **CNN-AE Model**: A convolutional autoencoder that denoises multispectral images.
- **Evaluation Metrics**: Performance evaluation using PSNR, RMSE, and Pearson Correlation Coefficient.
- **Simple Workflow**: End-to-end pipeline for training, evaluation, and testing with example scripts.
  
## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
   - Prepare the Dataset
   - Train the Model
   - Evaluate Results

## Installation
If using Conda, create an environment as follows:
conda create -n cnn_ae_env python=3.7
conda activate cnn_ae_env
pip install -r requirements.txt

### Project Structure

The project is organized as follows:

- CNN_AE_Model.py

- Main file containing the Convolutional Autoencoder model architecture, training loop, and evaluation procedures.
CNN_AE_Model.py
- Script for training the model. It loads the dataset, initializes the model, and begins the training process and  It calculates metrics like PSNR, RMSE, Pearson Correlation Coefficient and Registration Time to assess the model's performance.

Directory for storing the dataset. This is where you should place the raw dataset images.
Example subdirectories:
```Datase/
--600nm/
--620nm/
--670nm/
--760nm/

## Usage

### Prepare the Dataset
The dataset for this project contains multispectral images at different wavelengths. For example, images at 600nm, 620, 670nm, and 760nm wavelengths are stored in separate directories. The dataset is private, and access can be obtained by contacting the author at zhangtao@tju.edu.cn.
```/dataset/
600nm/
620nm/
670nm/
760nm/

### Train the Model
To train the Convolutional Autoencoder (CNN-AE) model, run the CNN_AE_Model.py script in your Python environment. This script includes both the model architecture and the training loop.
To start the training, run:
python CNN_AE_Model.py

After training the model, you will get the evaluation performance using metrics like PSNR, RMSE, and Pearson Correlation Coefficient, and Registration Time. 

### Requirements

The project requires Python 3.7 or later. Install dependencies from the `requirements.txt`:

```bash
pip install -r requirements.txt

Dependencies
Python 3.7+
PyTorch for model building and training
torchvision for image dataset handling
scikit-image, numpy, matplotlib for image processing and metrics
scipy for advanced calculations


### Summary:

1. **Installation**: Provides instructions for setting up the environment, including installing dependencies.
2. **Usage**: Explains how to prepare the dataset, train the model, and evaluate its performance.
3. **Project Structure**: A breakdown of how your project is organized.
4. **Evaluation Metrics**: Describes the metrics used to evaluate model performance.
5. **License**: A placeholder for the license you choose for your project.
6. **Acknowledgments**: I would like to express my sincere gratitude to Tianjin University for providing the support and resources necessary to complete this project. Their contribution was instrumental in the successful development of this work..



