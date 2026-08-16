import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.dataset import IMG_SIZE, preprocess_input
from src.evaluate import load_trained_model

def load_image(img_path):
    """
    Load and resize an image.

    Returns:
        np.ndarray:
            Image as a NumPy array.
    """
    image = Image.open(img_path).convert("RGB")
    image=image.resize((160,160))
    image_array = np.array(image)
    image_array = np.expand_dims(image, axis=0)
    return image_array

def predict_image(model_path,img_path):
    
    
    
    """
    Predict whether an image contains a cat or a dog.


    Returns:
        tuple:
            Predicted class and confidence score.
    """
    model=load_trained_model(model_path)
    image_array=load_image(img_path)
    plt.imshow(image_array.reshape(160,160,3))
    plt.axis("off")    
    plt.show()

    probability = model.predict(image_array, verbose=0)[0][0]

    if probability >= 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability

    return {
    "label": label,
    "confidence": float(confidence),
    "probability": float(probability)
    }