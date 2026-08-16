from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets


BATCH_SIZE = 32
# MOBILENET EXPECTS IMG SIZE TO BE AS FOLLOWING:
IMG_SIZE = (160, 160)
ROOT_DIR = Path(__file__).resolve().parents[1]


def create_dataset(train_dir=None, validation_dir=None):
    """
    Load the training and validation datasets from the project directories.

    The images are loaded using TensorFlow's
    `image_dataset_from_directory` function and returned as
    `tf.data.Dataset` objects.
    
    """
    
    train_dir = Path(train_dir) if train_dir else ROOT_DIR / "datasets" / "train"
    validation_dir = Path(validation_dir) if validation_dir else ROOT_DIR / "datasets" / "validation"
    
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            )
            
        ]
    )
    
    validation_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                #Values of mean and std is the the values eof the weights of the pretrained model
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225))
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_transform
    )

    validation_dataset  = datasets.ImageFolder(
        root=str(validation_dir),
        transform=validation_transform
    )

    
    class_names=train_dataset.classes
    
    # train_dataset, validation_dataset = prefetch(train_dataset, validation_dataset)

    return train_dataset, validation_dataset, class_names


def load_dataset():
    train_dataset, validation_dataset, class_names=create_dataset()
    
    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        
    )
    validation_loader=DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        
    )
    return train_loader, validation_loader, class_names
    

# def prefetch(train_dataset, validation_dataset):
#     """
#     Prefetch training and validation datasets to improve pipeline performance.
#     """

#     AUTOTUNE = tf.data.AUTOTUNE

#     train_dataset = train_dataset.prefetch(AUTOTUNE)
#     validation_dataset = validation_dataset.prefetch(AUTOTUNE)

#     return train_dataset, validation_dataset

# def data_augmenter():

    
#       data_augmentation=tf.keras.Sequential(
#             [
#                   RandomFlip("horizontal"),
#                   RandomRotation(0.2)
#             ]
#       )
#       return data_augmentation

# def preprocess_input():
#       """
#        Return the MobileNetV2 preprocessing function.
#       """
#       preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#       return preprocess_input