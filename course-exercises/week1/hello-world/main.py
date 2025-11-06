import tensorflow as tf
import numpy as np

def main():
    # Declare model inputs and outputs for training
    xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # Build a simple Sequential model
    model = tf.keras.Sequential([
        # Define the input shape
        tf.keras.Input(shape=(1,)),

        # Add a Dense layer
        tf.keras.layers.Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train the model
    model.fit(xs, ys, epochs=500)

    # Make a prediction
    print(f"model predicted: {model.predict(np.array([10.0]), verbose=0).item():.5f}")

if __name__ == "__main__":
    main()