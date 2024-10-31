Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using CNN-AE
## Overview
This project implements a Convolutional Autoencoder (CAE) to denoise images for improved visual clarity. 
The model learns to reconstruct clean images from noisy ones and is tailored for applications like early breast cancer screening. 
The project provides an end-to-end pipeline for training and evaluating the model with quality metrics, including Peak Signal-to-Noise Ratio (PSNR), Root Mean Square Error (RMSE), and Pearson Correlation Coefficient.

Features
Convolutional Autoencoder: Utilizes an encoder-decoder structure to denoise images.
the model learns to reconstruct the clean versions.
Evaluation Metrics: PSNR, RMSE, and Pearson Correlation Coefficient assess model performance.
Ease of Use: Simple training and evaluation scripts allow for reproducible results.
Easy-to-run training and evaluation scripts.


## Table of Contents
Installation
Project Structure
Usage
1. Prepare the Dataset
2. Train the Model
3. Results

## Installation

### Requirements

The project requires Python 3.7 or later. Install the dependencies listed in requirements.txt:

To set up and run this project, ensure you have Python installed. The required libraries are listed in the `requirements.txt` file. Install the dependencies by running:

```bash
pip install -r requirements.txt


This project has the following requirements.
Requirements
Python 3.7 or later
PyTorch
torchvision
skimage (scikit-image)
numpy
matplotlib

# For Conda environments, use:

conda create -n cnn_ae_env python=3.7
conda activate cnn_ae_env
pip install -r requirements.txt

# Dependencies:

Python 3.7+
PyTorch (for model building)
torchvision (for dataset handling)
scikit-image, numpy (for data processing)
matplotlib, scipy (for visualizations and metrics)

# Project Structure
CNN_AE_Model.py: Main file with model architecture and training loop and evaluating image quality.
conv_gray_img.py: Preprocesses images to grayscale.

# Prepare the Dataset
The dataset is private. Contact the author at zhangtao@tju.edu.cn for access. as we have 4 differernt wavelength images and we have to set the directories as every wavelength has the same wavelentgh images like 600nm images willbe plece in different directory and son on and then give the path one by one for training process and train the CNN_AE model individually for every directory and get the results as an evaluation metrics and also save the densoised processed clean images in the output direcotries. 
Arrange it as:

├── data
│   └── train
│       ├── noisy_images/
│       └── clean_images/


# Train the Model
Configure parameters in CNN_AE_Model.py (e.g., batch size, learning rate, and epochs).

Results
Metrics like PSNR and RMSE validate the reconstruction for image enhancement quality.

Note: As the dataset is private and experimental, please contact the corresponding author at zhangtao@tju.edu.cn for permission to use it. The dataset will be provided on request.
