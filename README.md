# Flower Classification with PyTorch

This project is a simple yet complete image classification pipeline to identify different types of flowers using PyTorch.

## Features

- Clean and split custom flower datasets
- Train a CNN model using PyTorch
- Evaluate model performance
- Predict flower type from a single image
- Optional GUI (PyQt5) for user-friendly prediction
- ONNX export for optimized deployment

## File Structure

├── data_get.py            # Download or collect image dataset  
├── data_clean.py          # Remove corrupted files, format organization  
├── data_split.py          # Split dataset into train/val/test  
├── train.py               # Train a CNN (e.g., ResNet)  
├── test.py                # Evaluate the trained model  
├── predict.py             # Predict flower class from image input  
├── onnxruntime_demo.py    # Run inference using ONNX  
├── window.py              # GUI for image upload and prediction  
├── torchutils.py          # Training helpers and utilities  
├── README.md              # Project documentation  

## Quick Start

1. (Optional) Create and activate conda env  
   conda create -n flower-env python=3.9  
   conda activate flower-env  

2. Install dependencies  
   pip install -r requirements.txt  

3. Prepare data  
   python data_get.py  
   python data_clean.py  
   python data_split.py  

4. Train model  
   python train.py  

5. Predict  
   python predict.py --image path_to_flower.jpg  

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- PyQt5 (for GUI)
- onnx, onnxruntime (for deployment)

## Notes

- You can modify the number of classes by changing the dataset folder structure.
- The output class count is based on the number of subfolders in your training data.

Built by Harry Zhang
