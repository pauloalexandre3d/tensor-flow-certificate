import numpy as np
import tensorflow as tf

# import unittests

def main():
    print("Hello, Housing Prices!")
    # Declare model inputs and outputs for training
    n_bedrooms = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    price_in_hundreds_of_thousands = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    print(f"Features have shape: {n_bedrooms.shape}")
    print(f"Targets have shape: {price_in_hundreds_of_thousands.shape}")

    # Build a simple Sequential model
    model = tf.keras.Sequential([
        # Define the input shape
        tf.keras.Input(shape=(1,)),

        # Add a Dense layer
        tf.keras.layers.Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    print("Model compiled.")

    # Train the model
    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500)
    print("Model trained.")

    new_n_bedrooms = np.array([7.0])
    predicted_price = model.predict(new_n_bedrooms, verbose=False).item()
    print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")

if __name__ == "__main__":
    main()