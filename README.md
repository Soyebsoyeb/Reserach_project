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
- 

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


<img width="977" alt="Screenshot 2025-07-08 at 10 17 13 PM" src="https://github.com/user-attachments/assets/d130598c-16ae-493a-96cb-55718d5a5d76" />








WITH SELF-ATTENTION:->

This project implements a deep learning solution for classifying maize (corn) leaf diseases using a modified VGG16 architecture with self-attention. The model achieved excellent performance, with 95% accuracy on the test set.


Model Architecture:->

The system uses a VGG16 backbone with several key modifications:
Self-Attention Mechanism: Added after the last convolutional layer to help the model focus on relevant disease patterns
Transfer Learning: Pretrained on ImageNet with frozen feature extractor layers
Custom Classifier: Modified fully connected layers with dropout for regularization


Training Process

Training Duration: 17 epochs (early stopping triggered)
Optimizer: Adam with learning rate 0.001
Learning Rate Scheduling: ReduceLROnPlateau with patience=2
Early Stopping: Patience=5 epochs
Data Augmentation: Random rotations, flips, color jitter, and Gaussian blur


Key Observations

The model performs exceptionally well on healthy leaves and common rust detection
Blight detection achieves good balance between precision and recall
No significant overfitting despite high training accuracy (validation metrics remain strong)



Technical Highlights
Implemented self-attention mechanism to improve feature selection
Used class weighting to handle potential imbalances
Employed comprehensive data augmentation to improve generalization
Achieved 94.5% validation accuracy with early stopping




