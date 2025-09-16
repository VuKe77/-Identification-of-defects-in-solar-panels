# Identification of Defects in Solar Panels

This repository implements a deep learning model using ResNet architecture to identify defects in solar panels. The model is built with PyTorch and aims to automate the detection of common panel defects, enhancing maintenance efficiency in solar energy systems.

## 🧠 Overview

- **Model Architecture**: ResNet (Residual Networks)  
- **Framework**: PyTorch  
- **Purpose**: Detect defects such as micro-cracks, discoloration, and other anomalies in solar panels.  
- **Input**: Images of solar panels  
- **Output**: Classification of images into 'Defective' or 'Non-Defective' categories  

## 📁 Project Structure

- `data.py` – Data preprocessing and augmentation  
- `model.py` – Definition of the ResNet model  
- `train.py` – Training script  
- `trainer.py` – Training loop and evaluation metrics  
- `export_onnx.py` – Export model to ONNX format  
- `environment.yml` – Conda environment configuration  
- `images.zip` – Sample dataset  
- `data.csv` – Metadata for the dataset  
- `analyse_data.ipynb` – Jupyter notebook for data analysis  
- `losses.png` – Visualization of training losses  
- `F1_metrics.png` – F1 score metrics visualization  

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  
- OpenCV  
- ONNX  

### Installation

Clone this repository:

```bash
git clone https://github.com/VuKe77/-Identification-of-defects-in-solar-panels.git
cd -Identification-of-defects-in-solar-panels
