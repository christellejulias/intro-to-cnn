# LAB | Image Classification and Recognition with CNNs

## Introduction

We have just learned how to use Convolutional Neural Networks (CNNs) for image classification, so let's practice a bit more by implementing a simple image classification model using TensorFlow and Keras.

<br>

## Requirements

1. Fork this repo.
2. Clone this repo.

## Submission

Once you finish the assignment, submit a URL link to your pull request in the field on Student Portal.

<br>

## Instructions

To complete this lab, you will need to install TensorFlow and Keras. You can find the dataset for image classification at [Kaggle's CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10). Make sure to refer back to the lesson on CNNs for guidance on model architecture.

## Tasks

### Task 1 /Iteration 1: Data Preparation

1. Load the CIFAR-10 dataset.
2. Normalize the pixel values to be between 0 and 1.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

### Task 2 /Iteration 2: Build the CNN Model

1. Define a CNN architecture with convolutional, pooling, and fully connected layers.

```python
from tensorflow.keras import layers, models

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Task 3 /Iteration 3: Compile and Train the Model

1. Compile the model using an appropriate optimizer and loss function.
2. Train the model on the training data.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Task 4 /Iteration 4: Evaluate Model Performance

1. Evaluate your model on the test dataset and print the accuracy.

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```

### Task 5 /Iteration 5: Make Predictions

1. Use the trained model to make predictions on a few test images.
2. Visualize the results.

```python
import matplotlib.pyplot as plt
import numpy as np

predictions = model.predict(x_test)

# Visualize results for first 5 test images
for i in range(5):
    plt.imshow(x_test[i])
    plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i][0]}')
    plt.axis('off')
    plt.show()
```

Feel free to modify the architecture or parameters to see how it affects performance!

