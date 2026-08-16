import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.PyTorch_src.model import DEVICE, load_trained_model
from src.PyTorch_src.dataset import IMG_SIZE, validation_transform



def load_image(img_path):
    """
    Load and preprocess an image for the PyTorch model.

    Returns:
        torch.Tensor:
            Image tensor with shape [1, C, H, W].
    """
    image = Image.open(img_path).convert("RGB")
    
    image_tensor = validation_transform(image)
    
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_image(model_path,img_path):
    
    
    
    """
    Predict whether an image contains a cat or a dog.

    """
    model=load_trained_model(model_path)
    image = Image.open(img_path).convert("RGB")
    image_tensor = load_image(img_path)
    image_tensor = image_tensor.to(DEVICE)
    
    plt.imshow(image)
    plt.axis("off")    
    plt.show()

    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()

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