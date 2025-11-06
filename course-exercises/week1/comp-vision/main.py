import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_example(training_images, training_labels):
    # You can put between 0 to 59999 here
    index = 0

    # Set number of characters per row when printing
    np.set_printoptions(linewidth=320)

    # Print the label and image
    print(f'LABEL: {training_labels[index]}')
    print(f'\nIMAGE PIXEL ARRAY:\n\n{training_images[index]}\n\n')

    # Visualize the image using the default colormap (viridis)
    plt.imshow(training_images[index])
    plt.colorbar()
    plt.show()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.89): # Experiment with changing this value
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

def main():
    print("Hello, Computer Vision!")

    callbacks = myCallback()

    # Load the Fashion MNIST dataset
    fmnist = tf.keras.datasets.fashion_mnist
    # Load the training and test split of the Fashion MNIST dataset
    (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

    # plot_example(training_images, training_labels)

    # Normalize the pixel values of the train and test images
    training_images  = training_images / 255.0
    test_images = test_images / 255.0

    # Build the classification model
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(28,28)),
        tf.keras.layers.Flatten(), 
        # tf.keras.layers.Dense(128, activation=tf.nn.relu), 
        tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

    print("Training:\n")
    model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])


    # Evaluate the model on unseen data
    print("\nEvaluating on test set:\n")
    model.evaluate(test_images, test_labels)

    predictions = model.predict(test_images)
    print(f"\nTrue class for first image on test set: {test_labels[0]}\nProbability of each class:\n{predictions[0]}")

    print(test_labels[0])


if __name__ == "__main__":
    main()