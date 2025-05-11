# Flower Classification with PyTorch

This project is a simple yet complete image classification pipeline to identify 5 types of flowers using PyTorch.

## Features

- Load and split custom flower datasets
- Train a CNN model (ResNet18) with transfer learning
- Evaluate model performance on a validation set
- Predict flower type from a single image via command line
- Save and reuse trained model (`.pth` file)
- Optional GUI (PyQt5) and ONNX export (to be added)

## File Structure
```
├── data_split.py # Split dataset into train/val/test
├── train.py # Train the model and save weights
├── predict.py # Predict flower class from image input
├── flower_resnet18.pth # Trained model (generated after training)
├── flowers_5/ # Raw image folders for 5 flower classes
│ ├── daisy/
│ ├── dandelion/
│ ├── roses/
│ ├── sunflowers/
│ └── tulips/
├── data/ # Generated train/val/test folders after split
│ ├── train/
│ ├── val/
│ └── test/
└── README.md # Project documentation 
```

## Quick Start

1. (Optional) Create and activate a conda environment
   ```
   conda create -n flower-env python=3.9  
   conda activate flower-env
   ```

3. Install dependencies
   ``` 
   pip install torch torchvision matplotlib scikit-learn
   ```

5. Prepare dataset  
   Organize your image folders like this:
   ```
   flowers_5/
       ├── daisy/
       ├── dandelion/
       ├── roses/
       ├── sunflowers/
       └── tulips/
   ```
   
   Then run:
   ```
   python data_split.py
   ```  

7. Train model
   ```
   python train.py
   ```

9. Predict on a new image
   ``` 
   python predict.py --image path/to/your_flower.jpg
   ```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- scikit-learn
- matplotlib

## Notes

- You must create a `flowers_5/` directory manually with subfolders for each class.
- The model outputs 5 classes based on subfolder names.
- After training, a file named `flower_resnet18.pth` is saved for prediction use.

Built by Harry Zhang
