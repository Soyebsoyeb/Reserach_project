# 🌽 Maize Leaf Disease Classification

## 🌟 Project Overview

This project implements a transfer learning approach to classify maize leaf diseases into multiple categories:
- Healthy leaves
- Common rust
- Gray leaf spot
- Northern leaf blight

The system achieves **~95% accuracy** using a fine-tuned VGG16 model with class-weighted loss handling.



## ✨ Key Features

- **Automated Dataset Preparation**: Kaggle API integration for one-click dataset setup
- **Advanced Image Augmentation**: Gaussian blur, random rotation, and normalization
- **Class Imbalance Handling**: Computed class weights for loss function
- **Model Evaluation**: Comprehensive metrics including confusion matrix and classification report
- **GPU Acceleration**: Automatic CUDA detection and utilization



**Statistics:**
- Total images: ~3,000
- Training set: 60%
- Validation set: 20%
- Test set: 20%

## 🧠 Model Architecture

**Base Model:** VGG16 (pre-trained on ImageNet)

**Modifications:**
1. Frozen feature extractor layers
2. Custom classifier head:

Sequential(
(0): Linear(in_features=25088, out_features=4096)
(1): ReLU(inplace=True)
(2): Dropout(p=0.5)
(3): Linear(in_features=4096, out_features=4096)
(4): ReLU(inplace=True)
(5): Dropout(p=0.5)
(6): Linear(in_features=4096, out_features=4) # Custom layer
)


**Training Parameters:**
- Optimizer: SGD (momentum=0.9)
- Learning rate: 0.001
- Batch size: 16
- Epochs: 25
- Loss: Weighted CrossEntropyLoss


📊 Epoch 25/25, Loss: 0.1243, Train Acc: 96.72%, Val Acc: 95.15%

🔍 Evaluating on Test Set...
📈 Classification Report:
                    precision  recall  f1-score   support

         healthy       0.97      0.96      0.97       203
     common_rust       0.94      0.95      0.94       198
  gray_leaf_spot       0.95      0.94      0.94       192
northern_leaf_blight  0.96      0.97      0.96       207

      accuracy                           0.96       800
     macro avg       0.96      0.96      0.96       800
  weighted avg       0.96      0.96      0.96       800




  📈 Visualizations
Confusion Matrix:
https://images/confusion_matrix.png

Training Progress:
https://images/training_curves.png

file:///var/folders/mh/zyvgrgvd7hxg_1ds_4_3qklc0000gn/T/TemporaryItems/NSIRD_screencaptureui_pwj4Ik/Screenshot%202025-07-08%20at%2010.15.23%E2%80%AFPM.png



