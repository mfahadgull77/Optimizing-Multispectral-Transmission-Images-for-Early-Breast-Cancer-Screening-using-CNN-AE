# Denoising Autoencoder for Image Denoising

## Project Overview
This project implements a Convolutional Autoencoder (CAE) for denoising images. The model is trained on noisy images and reconstructs their clean versions. The project also calculates evaluation metrics such as Peak Signal-to-Noise Ratio (PSNR), Root Mean Square Error (RMSE), and Pearson Correlation Coefficient to assess the quality of the denoised images.

## Directory Structure

```bash
project/
│
├── data/                     # Directory to store input images
│   ├── train/                # Training images go here
│
├── models/                   # Model definitions
│   ├── CNN_AE_Model.py       # Autoencoder model code
│
├── output/                   # Directory to save results and denoised images
│
└── train.py                  # Script to run the training

input directory/: Directory containing clean images for training and testing.
output directory/: Directory where the output (denoised images and JSON files) will be saved.
denoising_autoencoder.py: The main script implements the autoencoder architecture, training loop, and evaluation.

Note: As the dataset is private and experimental, please contact the corresponding author at zhangtao@tju.edu.cn for permission to use it. The dataset will be provided on request.
