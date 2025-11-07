import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def display_predictions(model, rescale_layer, image_dir='images'):
    """Display images with their predictions"""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 5))
    if len(image_files) == 1:
        axes = [axes]
    
    for i, filename in enumerate(image_files):
        file_path = os.path.join(image_dir, filename)
        
        # Load and predict
        image = tf.keras.utils.load_img(file_path, target_size=(300, 300))
        image_array = tf.keras.utils.img_to_array(image)
        image_scaled = rescale_layer(image_array)
        image_expanded = np.expand_dims(image_scaled, axis=0)
        
        prediction = model.predict(image_expanded, verbose=0)[0][0]
        label = "Human" if prediction > 0.5 else "Horse"
        
        # Display image
        axes[i].imshow(image)
        axes[i].set_title(f"{filename}\n{label} ({prediction:.2f})")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()