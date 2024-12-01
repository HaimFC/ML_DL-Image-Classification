# ML_DL Assignment 3 - Explanation and Solution

## Overview of the Assignment
This assignment focuses on the implementation and evaluation of convolutional neural networks (CNNs) for image-based tasks. Specifically, it involves:
1. Designing two CNN architectures:
   - A standard CNN model for image classification.
   - A CNN model with a unique channel structure.
2. Training the models to classify images into "same" or "different" pairs.
3. Evaluating the performance using saved models.

### Objectives:
- Understand the architecture and implementation of CNNs.
- Work with PyTorch to define, train, and evaluate deep learning models.
- Experiment with modifications in CNN structures to observe their effects on performance.

---

## Solution Description

### 1. CNN Architecture
**Task:** Implement a standard CNN for binary classification tasks.  
**Solution:**  
The `CNN` class in `ML_DL_Functions3.py` defines the architecture:
- Convolutional layers extract features with increasing depth.
- Max pooling layers reduce spatial dimensions.
- Fully connected layers process flattened features for classification.
- Dropout is used to prevent overfitting.
- The output is passed through a log-softmax activation for class probabilities.

#### Architecture Details:
- Input shape: \( (N, 3, 448, 224) \) where \( N \) is the batch size.
- Output shape: \( (N, 2) \), indicating the probabilities for "same" and "different" classes.

---

### 2. CNNChannel Architecture
**Task:** Implement a CNN that processes images with a specific channel structure.  
**Solution:**  
The `CNNChannel` class introduces a modification:
- Splits the image into top and bottom halves.
- Combines them along the channel dimension to form a 6-channel input.
- Passes the 6-channel input through convolutional and pooling layers.
- Fully connected layers and dropout ensure classification accuracy.

#### Architecture Details:
- Input shape: \( (N, 3, 448, 224) \).
- After splitting and combining: \( (N, 6, 224, 224) \).
- Output shape: \( (N, 2) \), same as the `CNN`.

---

### 3. Pretrained Models
The assignment includes pretrained models saved as:
- `best_CNN_model.pk`: Contains the trained weights and biases for the standard CNN.
- `best_CNNChannel_model.pk`: Contains the trained weights and biases for the CNNChannel model.

These models can be loaded and evaluated to analyze performance.

---

### Functions in `ML_DL_Functions3.py`

#### **`ID1()`**
Returns the personal ID of the first student.

#### **`ID2()`**
Returns the personal ID of the second student (if applicable).

#### **`CNN` Class**
Implements the standard CNN architecture with the following layers:
1. **Convolutional Layers**: Extract features with 8, 16, 32, and 64 filters.
2. **Pooling Layers**: Max pooling to downsample features.
3. **Fully Connected Layers**: Dense layers for final classification.
4. **Dropout**: Regularization to reduce overfitting.

#### **`CNNChannel` Class**
Implements a CNN with channel-based modifications:
1. **Splitting Input**: Divides the input into top and bottom halves.
2. **Combining Channels**: Merges halves into a 6-channel tensor.
3. **Convolution and Pooling**: Similar to `CNN` but adapted for 6-channel input.

---

## Evaluation
1. **Loading Pretrained Models**:
   - Use PyTorch to load the `.pk` files containing the best-trained models.
2. **Testing on New Data**:
   - Pass test data through the models to evaluate their performance.
3. **Metrics**:
   - Accuracy and loss are used to compare the two architectures.

---

## Summary
This assignment demonstrates the design and implementation of CNN architectures. By experimenting with standard and modified structures, it explores how changes in input processing affect performance. The provided pretrained models validate the effectiveness of the implementations.
