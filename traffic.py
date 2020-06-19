import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    """
    images = []
    labels = []
    for category in os.listdir(data_dir):
        if not category.startswith('.'):
            for filename in os.listdir(os.path.join(data_dir, category)):
                filepath = os.path.join(data_dir, category, filename)
                im = cv2.imread(filepath)
                resized = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resized)
                labels.append(category)
    return (images, labels)            



def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = tf.keras.Sequential()
    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(30, 30, 3)))
    # Pooling Layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    # Flatten 3-D to 1-D
    model.add(tf.keras.layers.Flatten())
    # Hidden Layer
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # Output Layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    model.compile(optimizer="adam",
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
