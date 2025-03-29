# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.

## Dataset
The dataset used in this project is the CIFAR-10 dataset, which contains 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is loaded using `tensorflow.keras.datasets` and is split into training and testing sets.

## Prerequisites
To run this project, you need to have the following libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow`

You can install them using:
```bash
pip install pandas numpy matplotlib tensorflow
```

## Model Architecture
The CNN model consists of the following layers:
1. **Convolutional Layer 1**: 32 filters of size (3x3), ReLU activation
2. **Max Pooling Layer 1**: Pool size (2x2)
3. **Convolutional Layer 2**: 64 filters of size (3x3), ReLU activation
4. **Max Pooling Layer 2**: Pool size (2x2)
5. **Convolutional Layer 3**: 64 filters of size (3x3), ReLU activation
6. **Flatten Layer**: Converts the 2D feature maps into a 1D vector
7. **Dense Layer 1**: Fully connected layer with 64 neurons, ReLU activation
8. **Dense Layer 2 (Output Layer)**: 10 neurons (one per class), no activation function (logits output)

## Training
The model is compiled with:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy (with logits)
- **Metric**: Accuracy

It is trained for **10 epochs** using the training dataset.

## Evaluation
After training, the model is evaluated on the test dataset to compute accuracy.

## Prediction and Visualization
The model makes predictions on the test dataset. An example image from the test set is displayed along with its predicted class.

## Running the Script
To execute the script, simply run:
```bash
python script.py
```
Ensure that all dependencies are installed before running.

## Output
- Training loss and accuracy per epoch
- Test accuracy after evaluation
- The predicted label for a sample test image
- Display of the test image using Matplotlib

## Future Improvements
- Increase the number of epochs for better accuracy
- Implement data augmentation to improve generalization
- Tune hyperparameters (learning rate, batch size, etc.)
- Experiment with deeper architectures or different activation functions

## License
This project is open-source and free to use for educational purposes.
