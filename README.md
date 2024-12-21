# 20_Epoch
Building MNIST model to run on only 20K and less params & achieve 99% accuracy with 20 Epoch. 
Description This project demonstrates an innovative approach to training a machine learning model that achieves 99% accuracy on the MNIST dataset in 20 epochs. By leveraging efficient data preprocessing, model architecture optimization, and advanced initialization techniques, this project sets a new benchmark for speed and accuracy on MNIST. This model can serve as a foundation for tasks requiring rapid and resource-efficient training.

Table of Contents
## Introduction
This project implements a CNN model for classifying MNIST digits.
## Dataset
The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits.
## Model Architecture
The model is a simple CNN with fewer than 20,000 parameters.
## Training
To train the model, run the 20_epoch.py script. The script is configured to train the model on the MNIST dataset for one epoch by default, but you can customize parameters such as the number of epochs, batch size, and learning rate.
## Default params
Epochs: 20
Batch size: 64
Learning rate: 0.001
## Results
After just one epoch of training, the model achieves over 95% accuracy on the MNIST test dataset.
Performance Metrics:
Accuracy on Test Set: ~99%
Number of Parameters: ~19898 (Lightweight CNN model)
This lightweight model is optimized to work efficiently with minimal computational resources while maintaining high accuracy.
