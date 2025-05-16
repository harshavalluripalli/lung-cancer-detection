# Lung Cancer Detection using Deep Learning

![Project Banner](https://via.placeholder.com/800x200?text=Lung+Cancer+Detection+using+Deep+Learning)

A deep learning-based system for detecting and classifying lung cancer from medical images using various CNN architectures and transfer learning techniques.

## Table of Contents
- [Abstract](#abstract)
- [Features](#features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [References](#references)

## Abstract
Lung cancer is one of the leading causes of death worldwide, with approximately five million fatal cases annually. Early detection is challenging due to small nodule sizes and their locations. This project implements an automatic lung cancer detection system using deep learning to improve accuracy and reduce diagnosis time compared to existing methods.

The system processes CT scan images through:
1. Histogram Equalization for contrast enhancement
2. Threshold Segmentation for image simplification
3. Deep Learning models for classification

## Features
- Web interface (Flask) for easy interaction
- Gradio demo for quick testing
- Support for multiple deep learning architectures:
  - Custom CNN
  - VGG16/VGG19
  - MobileNet
  - ResNet50
  - Xception
  - InceptionV3
- Comprehensive preprocessing pipeline
- Performance metrics visualization

## Methodology
![Methodology Diagram](./methodology.jpeg)

1. **Data Preprocessing**:
   - Histogram Equalization for contrast enhancement
   - Threshold Segmentation for feature extraction

2. **Model Training**:
   - Transfer learning with multiple architectures
   - Custom CNN implementation
   - Data augmentation techniques

3. **Evaluation**:
   - Multiple metrics (Accuracy, Precision, Recall, AUC, F1-score)
   - Comparative analysis

## Installation

### Prerequisites
- Python 3.7+
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lung-cancer-detection.git
   cd lung-cancer-detection

   Install dependencies:

bash
pip install -r requirements.txt
Download the pre-trained models (if available) or train your own.

Usage
Web Interface (Flask)
bash
python app.py


Gradio Demo
bash
python gradio_app.py



