import tensorflow as tf
from tensorflow.keras import layers
from src.dataset import data_augmenter, IMG_SIZE, preprocess_input



IMG_SHAPE=IMG_SIZE + (3,)

def create_model(data_augmentation=data_augmenter()):
    """
    Build the Cats vs Dogs classifier using transfer learning.

    A pretrained MobileNetV2 model is used as the feature extractor
    with frozen weights. A custom classification head is added on top.
    Returns:
    tf.keras.Model:
    Binary image classification model.
    """

    base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SHAPE
    )

    base_model.trainable=False
    inputs=tf.keras.Input(shape=IMG_SHAPE)
    x=data_augmentation(inputs)
    x=preprocess_input()(x)
    x=base_model(x,training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.2)(x)

    outputs=layers.Dense(1, activation='sigmoid')(x)
    model=tf.keras.Model(inputs, outputs)

    return model

