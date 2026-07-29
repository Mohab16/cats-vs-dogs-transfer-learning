import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers
from src.model import create_model
from src.dataset import load_dataset

def get_callbacks(FILE_PATH):
    callbacks=[tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
    filepath=FILE_PATH,
    mode="min",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
    )

    ]
    return callbacks



def compile_model(model, learning_rate):
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss="binary_crossentropy",
    metrics=['accuracy']
    )

def train_head_model():
    """
    Train only the custom classification head.

    During this stage, the pretrained MobileNetV2 backbone
    remains frozen while only the newly added classifier
    layers are updated.

    Returns:
        tuple:
            model (tf.keras.Model):
                Trained model.

            history (tf.keras.callbacks.History):
                Training history.
    """

    train_dataset, validation_dataset,_=load_dataset()
    
    model=create_model()
    compile_model(model, 0.001)
    head_history=model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=get_callbacks("../models/head_training.keras"))
    return model,head_history

def fine_tune_model(initial_epoch):
    """
    Fine-tune the pretrained MobileNetV2 backbone.

    The pretrained model is loaded from disk, the top layers
    are unfrozen, and training resumes using a smaller
    learning rate.

    Args:
        initial_epoch (int):
            Epoch from which fine-tuning should continue.

    Returns:
        tuple:
            model (tf.keras.Model):
                Fine-tuned model.

            history (tf.keras.callbacks.History):
                Fine-tuning history.
    """
    model=tf.keras.models.load_model('../models/head_training.keras')
    base_model=model.get_layer("mobilenetv2_1.00_160")
    train_dataset, validation_dataset,_=load_dataset()
    
    base_model.trainable=True
    fine_tune_at=120

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable=False
    compile_model(model, 0.00005)
    final_history=model.fit(train_dataset, epochs=10, initial_epoch=initial_epoch,validation_data=validation_dataset,callbacks=get_callbacks("../models/final_training.keras"))
    return model, final_history


