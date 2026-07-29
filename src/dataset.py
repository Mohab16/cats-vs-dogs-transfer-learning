from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation

BATCH_SIZE = 32
# MOBILENET EXPECTS IMG SIZE TO BE AS FOLLOWING:
IMG_SIZE = (160, 160)
ROOT_DIR = Path(__file__).resolve().parents[1]


def load_dataset(train_dir=None, validation_dir=None):
    """
    Load the training and validation datasets from the project directories.

    The images are loaded using TensorFlow's
    `image_dataset_from_directory` function and returned as
    `tf.data.Dataset` objects.
    
    """
    
    train_dir = Path(train_dir) if train_dir else ROOT_DIR / "datasets" / "train"
    validation_dir = Path(validation_dir) if validation_dir else ROOT_DIR / "datasets" / "validation"

    train_dataset = image_dataset_from_directory(
        directory=str(train_dir),
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
    )

    validation_dataset = image_dataset_from_directory(
        directory=str(validation_dir),
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
    )
    
    class_names=train_dataset.class_names
    
    train_dataset, validation_dataset = prefetch(train_dataset, validation_dataset)

    return train_dataset, validation_dataset, class_names
    

def prefetch(train_dataset, validation_dataset):
    """
    Prefetch training and validation datasets to improve pipeline performance.
    """

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)

    return train_dataset, validation_dataset

def data_augmenter():

    
      data_augmentation=tf.keras.Sequential(
            [
                  RandomFlip("horizontal"),
                  RandomRotation(0.2)
            ]
      )
      return data_augmentation

def preprocess_input():
      """
       Return the MobileNetV2 preprocessing function.
      """
      preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
      return preprocess_input