# Handwriting Recognition

# Import TensorFlow
import tensorflow as tf

# Import Data Set from mnist library
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Shapes of Imported Arrays
print("x_Train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Plot an Image from the dataset
from matplotlib import pyplot as plt
%matplotlib inline
plt.imshow(x_train[0], cmap="binary")
plt.show()

# Displaying Training Dataset Labels
y_train[0]
print(set(y_train))

# One Hot Encoding
from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Validating Shapes of Encoded Labels
print("y_train_encoded", y_train_encoded)
print("y_test_encoded", y_test_encoded)

# Unrolling N-dimensional Array to Vectors
import numpy as np
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))
print("x_train_reshaped", x_train_reshaped)
print("x_test_reshaped", x_test_reshaped)

# Display Pixel Value
print(set(x_train_reshaped[0]))

# Data Normalization
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# Display Normalized Data
print(set(x_train_norm[0]))

# Creating a Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation="relu", input_shape=(784,)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Compiling the Model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the Model
model.fit(x_train_norm, y_train_encoded, epochs=3)

# Evaluating the Model
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print("Accuracy of model is", accuracy * 100)

# Predictions on Test Set
preds = model.predict(x_test_norm)
print("Shape of preds", preds)

# Plotting the Results
plt.figure(figsize=(12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index + i])
    gt = y_test[start_index + i]

    col = "g" if pred == gt else "r"
    plt.xlabel(f"i = {start_index + i}, pred = {pred}, gt = {gt}", color=col)
    plt.imshow(x_test[start_index + i], cmap="binary")

plt.show()

# Plot the prediction for a single test example
plt.plot(preds[8])
plt.show()