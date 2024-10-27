# Denoising Autoencoder for Image Denoising

## Overview

This project implements a Convolutional Autoencoder (CAE) to denoise images. The model is trained on noisy images and reconstructs their clean counterparts. The project provides an easy-to-follow pipeline, from training to evaluating the model using popular image-quality metrics like Peak Signal-to-Noise Ratio (PSNR), Root Mean Square Error (RMSE), and Pearson Correlation Coefficient.

## Features

- Convolutional Autoencoder architecture.
- Gaussian noise is added to the clean images, and the model learns to reconstruct the clean version.
- Evaluation metrics include PSNR, RMSE, and Pearson Correlation Coefficient.
- Easy-to-run training and evaluation scripts.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Prepare the Dataset](#1-prepare-the-dataset)
  - [2. Train the Model](#2-train-the-model)
  - [3. Evaluate the Model](#3-evaluate-the-model)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Installation

### Requirements

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


denoising_autoencoder.py: The main script implements the autoencoder architecture, training loop, and evaluation.

Note: As the dataset is private and experimental, please contact the corresponding author at zhangtao@tju.edu.cn for permission to use it. The dataset will be provided on request.
