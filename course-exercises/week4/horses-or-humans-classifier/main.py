import os
import random

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np

# Plotting and dealing with images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

# Import display function
from display_images import display_predictions

from intermediate_representaion import show_intermediate_representation

TRAIN_DIR = 'horse-or-human'


def main():
    # You should see a `horse-or-human` folder here
    print(f"files in current directory: {os.listdir()}")

    # Check the subdirectories
    print(f"\nsubdirectories within '{TRAIN_DIR}' dir: {os.listdir(TRAIN_DIR)}")

    # Directory with the training horse pictures
    train_horse_dir = os.path.join(TRAIN_DIR, 'horses')

    # Directory with the training human pictures
    train_human_dir = os.path.join(TRAIN_DIR, 'humans')

    # Check the filenames
    train_horse_names = os.listdir(train_horse_dir)
    print(f"5 files in horses subdir: {train_horse_names[:5]}")
    train_human_names = os.listdir(train_human_dir)
    print(f"5 files in humans subdir:{train_human_names[:5]}")

    print(f"total training horse images: {len(os.listdir(train_horse_dir))}")
    print(f"total training human images: {len(os.listdir(train_human_dir))}")

    plot_training_examples(train_horse_dir, train_horse_names, train_human_dir, train_human_names)

    model = build_model()

    train_dataset = create_training_dataset()

    # Get one batch from the dataset
    sample_batch = list(train_dataset.take(1))[0]

    # Check that the output is a pair
    print(f'sample batch data type: {type(sample_batch)}')
    print(f'number of elements: {len(sample_batch)}')

    # Extract image and label
    image_batch = sample_batch[0]
    label_batch = sample_batch[1]

    # Check the shapes
    print(f'image batch shape: {image_batch.shape}')
    print(f'label batch shape: {label_batch.shape}')

    # You can also preview the image array so you can compare the pixel values later in the next step of the preprocessing.
    print(image_batch[0].numpy())

    # Check the range of values
    print(f'max value: {np.max(image_batch[0].numpy())}')
    print(f'min value: {np.min(image_batch[0].numpy())}')

    rescale_layer = tf.keras.layers.Rescaling(scale=1. / 255)
    image_scaled = rescale_layer(image_batch[0]).numpy()
    print(image_scaled)
    print(f'max value: {np.max(image_scaled)}')
    print(f'min value: {np.min(image_scaled)}')

    # Rescale the image using a lambda function
    train_dataset_scaled = train_dataset.map(lambda image, label: (rescale_layer(image), label))

    # # Same result as above but without using a lambda function
    # # define a function to rescale the image
    # def rescale_image(image, label):
    #     return rescale_layer(image), label

    # dataset_scaled = dataset.map(rescale_image)

    # Get one batch of data
    sample_batch = list(train_dataset_scaled.take(1))[0]

    # Get the image
    image_scaled = sample_batch[0][1].numpy()

    # Check the range of values for this image
    print(f'max value: {np.max(image_scaled)}')
    print(f'min value: {np.min(image_scaled)}')

    # tuning the training dataset
    SHUFFLE_BUFFER_SIZE = 1000
    PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

    train_dataset_final = (train_dataset_scaled
                           .cache()
                           .shuffle(SHUFFLE_BUFFER_SIZE)
                           .prefetch(PREFETCH_BUFFER_SIZE)
                           )

    history = model.fit(
        train_dataset_final,
        epochs=15,
        verbose=2
    )

    plot_accuracy(history)

    display_predictions(model, rescale_layer)

    show_intermediate_representation(model, rescale_layer,
                                     train_horse_dir, train_horse_names,
                                     train_human_dir, train_human_names)


def plot_training_examples(train_horse_dir, train_horse_names, train_human_dir, train_human_names):
    # Parameters for your graph; you will output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 3, nrows * 3)

    next_horse_pix = [os.path.join(train_horse_dir, fname)
                      for fname in random.sample(train_horse_names, k=8)]
    next_human_pix = [os.path.join(train_human_dir, fname)
                      for fname in random.sample(train_human_names, k=8)]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


def build_model():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.Input(shape=(300, 300, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0 to 1 where 0 is for 'horses' and 1 for 'humans'
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def create_training_dataset():
    # Instantiate the dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(300, 300),
        batch_size=32,
        label_mode='binary'
    )

    # Check the type
    dataset_type = type(train_dataset)
    print(f'train_dataset inherits from tf.data.Dataset: {issubclass(dataset_type, tf.data.Dataset)}')
    return train_dataset


def plot_accuracy(history):
    # Plot the training accuracy for each epoch
    acc = history.history['accuracy']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    main()
