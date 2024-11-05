# Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using CNN-AE

This project implements a Convolutional Autoencoder (CNN-AE) to denoise multispectral transmission images, which is crucial for early breast cancer screening. The model learns to reconstruct clean images from noisy multispectral images, improving image clarity for accurate analysis.

## Features
- **CNN-AE Model**: A convolutional autoencoder that denoises multispectral images.
- **Evaluation Metrics**: Performance evaluation using PSNR, RMSE, and Pearson Correlation Coefficient.
- **Simple Workflow**: End-to-end pipeline for training, evaluation, and testing with example scripts.
  
## Project Structure
CNN_AE_Model.py: The core model, architecture, training loop, and evaluation.
conv_gray_img.py: Preprocesses the images, including grayscale conversion.

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

## Prepare the Dataset
The dataset for this project contains multispectral images at different wavelengths. For example, images at 600nm, 620, 670nm, and 760nm wavelengths are stored in separate directories. The dataset is private, and access can be obtained by contacting the author at zhangtao@tju.edu.cn.
/dataset/
  ├── 600nm/
  ├── 620nm/
  ├── 670nm/
  ├── 700nm/




