import matplotlib.pyplot as plt
import torch
from src.PyTorch_src.model import CatsDogsModel

import numpy as np

DEVICE = torch.device(
    "cuda" if torch.cudnn_is_available() else "cpu"
)

def load_trained_model(model_path):

    trained_model = CatsDogsModel()
    trained_model.load_state_dict(
        torch.load(
            model_path,
            map_location= DEVICE
        )
    )
    
    trained_model = trained_model.to(DEVICE)
    
    trained_model.eval()
    
    return trained_model

def evaluate_model(model_path, validation_dataset):
    """
    Evaluate a trained model.

    Returns:
        tuple:
            Validation loss and accuracy.
    """
    model=load_trained_model(model_path)
    loss, accuracy=model.evaluate(validation_dataset)
    return loss, accuracy

def predict_validation_set(model_path):
    _, validation_dataset=load_dataset()
    model=load_trained_model(model_path)
    y_pred=model.predict(validation_dataset)
    return y_pred
    
    
def plot_history(history):
    """
    Plot training and validation accuracy and loss.

    """

    history = history.history

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_confusion_matrix(model_path, validation_dataset):
    """
    Plot confusion matrix for the validation dataset.
    """

    model = load_trained_model(model_path)

    y_true = []
    y_pred = []

    for images, labels in validation_dataset:
        predictions = model.predict(images)

        # Binary classification
        predictions = (predictions > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.flatten())

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Cat", "Dog"]
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.show()    
