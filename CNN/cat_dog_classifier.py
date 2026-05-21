import tensorflow as tf  # # Imports TensorFlow, the main deep learning library
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # # For loading and augmenting images
import matplotlib.pyplot as plt  # # For plotting training results
import os

# # Set directory paths for training and testing datasets
train_dir = "Dogs-vs-Cats/training_set/training_set"
test_dir = "Dogs-vs-Cats/test_set/test_set"

# # Create data generators for loading images and applying augmentations
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# # Load and preprocess training images in batches from directory
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=32, class_mode="binary"
)

# # Load and preprocess testing images in batches from directory
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(64, 64), batch_size=32, class_mode="binary"
)

# # Define a simple Convolutional Neural Network (CNN) model for image classification
model = tf.keras.models.Sequential(
    [
        # Input layer with 32 filters, 3x3 kernel, and ReLU activation # # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        # # Max pooling layer to reduce spatial dimensions  # # First max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # # Second convolutional layer with 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # # Second max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # # Flatten the output to feed into fully connected layers  # # Flatten to 1D vector
        tf.keras.layers.Flatten(),
        # # Fully connected layer with 128 neurons and ReLU activation  # # Fully connected dense layer
        tf.keras.layers.Dense(128, activation="relu"),
        # Binary output: cat or dog # # Output layer (sigmoid for binary
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# # Compile the model with optimizer, loss function, and metrics
model.summary()  # Print model summary to see the architecture # # Compile the model with optimizer, loss function, and metric
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Train the model using the training data generator and validate on the test data generator
history = model.fit(train_generator, epochs=10, validation_data=test_generator)
# # Save the trained model to a file for later use
model.save("cat_dog_classifier.h5")

# Plot accuracy and loss
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# import numpy as np
# from tensorflow.keras.preprocessing import image
#
# img_path = 'Dogs-vs-Cats/test_set/test_set/cats/cat.1.jpg'     # # Path to image for prediction
# img = image.load_img(img_path, target_size=(64, 64))           # # Load and resize image
# img_array = image.img_to_array(img) / 255.0                    # # Convert image to array and normalize pixel values
# img_array = np.expand_dims(img_array, axis=0)                  # # Add batch dimension for prediction
#
# prediction = model.predict(img_array)
# if prediction[0][0] > 0.5:
#    print("It's a dog!")
# else:
#    print("It's a cat!")
#
